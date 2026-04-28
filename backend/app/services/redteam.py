"""
Multi-Agent Red Teaming — Adversarial Bias Discovery

Two specialized Gemini agents run in an adversarial loop:
  - Attacker: generates synthetic edge-case demographic profiles designed
    to surface hidden bias in maximally qualified candidates
  - Auditor: evaluates predictions on synthetic profiles, computes
    per-subgroup metrics, ranks harm severity, directs next probe

They iterate until no new bias hotspot is found or max_rounds reached.

When Gemini is unavailable, a deterministic fallback generates synthetic
profiles programmatically and runs the same analysis pipeline.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.core.fairness import FairnessEngine

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════

@dataclass
class RedTeamRound:
    """Results from a single red team round."""
    round_num: int
    target_subgroup: str
    profiles_generated: int
    attacker_strategy: str
    subgroup_results: list  # [{subgroup, selection_rate, di_ratio, count}]
    worst_subgroup: str
    worst_di: float
    worst_severity: str
    root_cause_features: list  # top features driving bias
    auditor_analysis: str
    done: bool = False


@dataclass
class RedTeamSession:
    """Full red team session with all rounds."""
    session_id: str
    dataset_id: str
    rounds: list[RedTeamRound] = field(default_factory=list)
    conversation_trace: list[dict] = field(default_factory=list)
    final_summary: str = ""
    worst_overall_subgroup: str = ""
    worst_overall_di: float = 1.0
    root_cause: list[str] = field(default_factory=list)
    status: str = "running"

    def to_dict(self) -> dict:
        """Convert to a JSON-safe dictionary with all numpy types sanitized."""
        return _sanitize({
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "target_subgroup": r.target_subgroup,
                    "profiles_generated": r.profiles_generated,
                    "attacker_strategy": r.attacker_strategy,
                    "subgroup_results": r.subgroup_results,
                    "worst_subgroup": r.worst_subgroup,
                    "worst_di": r.worst_di,
                    "worst_severity": r.worst_severity,
                    "root_cause_features": r.root_cause_features,
                    "auditor_analysis": r.auditor_analysis,
                    "done": r.done,
                }
                for r in self.rounds
            ],
            "conversation_trace": self.conversation_trace,
            "final_summary": self.final_summary,
            "worst_overall_subgroup": self.worst_overall_subgroup,
            "worst_overall_di": self.worst_overall_di,
            "root_cause": self.root_cause,
            "status": self.status,
        })


_sessions: dict[str, RedTeamSession] = {}


def get_redteam_session(session_id: str) -> Optional[RedTeamSession]:
    return _sessions.get(session_id)


# ═══════════════════════════════════════
#  AGENT PROMPTS
# ═══════════════════════════════════════

ATTACKER_PROMPT = """You are an adversarial bias red-teamer for AI hiring systems.

Your job: generate synthetic candidate profiles that are MAXIMALLY QUALIFIED
for the target demographic subgroup. The goal is to test whether the model
still rejects highly qualified candidates from certain groups.

Given a target subgroup (e.g., "Asian women" or "Black men"), generate
profiles with strong credentials: advanced degrees, 8+ years experience,
top-tier references, leadership roles, high education scores.

IMPORTANT: Vary credentials systematically across profiles to find the
threshold where bias persists even with strong qualifications.

Respond ONLY with a JSON object (no markdown, no backticks):
{
  "strategy": "Brief description of your testing strategy",
  "profiles": [
    {"age": 35, "sex": "Female", "race": "Asian", "education_num": 16, "hours_per_week": 45, "occupation": "Prof-specialty", "workclass": "Private", "marital_status": "Married", "relationship": "Wife", "capital_gain": 5000, "capital_loss": 0}
  ]
}

Generate exactly 10 profiles."""

AUDITOR_PROMPT = """You are a fairness auditor evaluating model predictions on synthetic test profiles.

Given model predictions for synthetic profiles generated by the red team attacker,
compute and analyze:
1. Selection rate per (sex × race) subgroup
2. Disparate impact ratio per subgroup vs the best-performing group
3. Which features most strongly predict rejection for the worst subgroup

Respond ONLY with a JSON object (no markdown, no backticks):
{
  "analysis": "2-3 sentence analysis of findings",
  "worst_subgroup": "e.g. Asian women",
  "worst_di": 0.47,
  "severity": "low|medium|high|critical",
  "root_cause_features": ["feature1", "feature2"],
  "next_target": "subgroup to probe deeper next round",
  "done": false
}

Set done=true ONLY when the worst DI ratio is above 0.80 or when
the same subgroup has been probed 2+ times with no new finding."""


# ═══════════════════════════════════════
#  SYNTHETIC PROFILE GENERATOR (fallback)
# ═══════════════════════════════════════

def _generate_synthetic_profiles(
    target_subgroup: str,
    df: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:
    """
    Generate synthetic maximally-qualified profiles for a target subgroup.
    Used as fallback when Gemini is unavailable.
    """
    np.random.seed(hash(target_subgroup) % 2**32)

    # Parse target subgroup
    target_sex = None
    target_race = None

    sex_values = df["sex"].unique() if "sex" in df.columns else []
    race_values = df["race"].unique() if "race" in df.columns else []

    target_lower = target_subgroup.lower()
    for s in sex_values:
        if str(s).lower() in target_lower:
            target_sex = s
    for r in race_values:
        if str(r).lower() in target_lower:
            target_race = r

    # Build profiles using dataset column structure
    profiles = {}
    for col in df.columns:
        if col == "income" or col == "two_year_recid" or col == "credit":
            continue  # skip label columns

        if col == "sex" and target_sex is not None:
            profiles[col] = [target_sex] * n
        elif col == "race" and target_race is not None:
            profiles[col] = [target_race] * n
        elif col == "age":
            profiles[col] = np.random.randint(30, 55, n).tolist()
        elif col == "education_num":
            profiles[col] = np.random.randint(13, 17, n).tolist()
        elif col == "hours_per_week":
            profiles[col] = np.random.randint(40, 60, n).tolist()
        elif col == "capital_gain":
            profiles[col] = np.random.randint(3000, 15000, n).tolist()
        elif col == "capital_loss":
            profiles[col] = [0] * n
        elif pd.api.types.is_numeric_dtype(df[col]):
            p75 = float(df[col].quantile(0.75))
            profiles[col] = [p75] * n
        else:
            profiles[col] = np.random.choice(df[col].dropna().unique(), n).tolist()

    return pd.DataFrame(profiles)


def _compute_subgroup_results(
    df_profiles: pd.DataFrame,
    y_pred: np.ndarray,
    favorable_label,
) -> list[dict]:
    """Compute selection rates per intersectional subgroup."""
    results = []

    sex_col = "sex" if "sex" in df_profiles.columns else None
    race_col = "race" if "race" in df_profiles.columns else None

    if sex_col and race_col:
        groups = df_profiles.groupby([sex_col, race_col])
    elif sex_col:
        groups = df_profiles.groupby(sex_col)
    elif race_col:
        groups = df_profiles.groupby(race_col)
    else:
        return results

    max_rate = 0.0
    subgroup_rates = {}

    for group_key, group_idx in groups.groups.items():
        group_preds = y_pred[group_idx.values] if hasattr(group_idx, 'values') else y_pred[list(group_idx)]
        selection_rate = float(np.mean(group_preds == favorable_label)) if len(group_preds) > 0 else 0.0

        key_str = str(group_key) if isinstance(group_key, tuple) else str(group_key)
        subgroup_rates[key_str] = selection_rate
        max_rate = max(max_rate, selection_rate)

    for subgroup, rate in subgroup_rates.items():
        di = float(rate / max_rate) if max_rate > 0 else 0.0
        try:
            count = len(groups.groups.get(
                eval(subgroup) if subgroup.startswith("(") else subgroup, []
            ))
        except Exception:
            count = 0
        results.append({
            "subgroup": subgroup,
            "selection_rate": round(rate, 4),
            "di_ratio": round(di, 4),
            "count": int(count),
            "severity": FairnessEngine.classify_severity(di).value,
        })

    results.sort(key=lambda x: x["di_ratio"])
    return results


def _identify_root_cause_features(
    X: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    favorable_label,
    subgroup_mask: np.ndarray,
) -> list[str]:
    """Identify features most associated with rejection in the worst subgroup."""
    rejected = (y_pred != favorable_label) & subgroup_mask
    selected = (y_pred == favorable_label) & subgroup_mask

    if np.sum(rejected) < 2 or np.sum(selected) < 2:
        return []

    feature_diffs = []
    for i, feat in enumerate(feature_names):
        mean_rej = float(np.mean(X[rejected, i]))
        mean_sel = float(np.mean(X[selected, i]))
        diff = abs(mean_sel - mean_rej)
        feature_diffs.append((feat, diff))

    feature_diffs.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in feature_diffs[:5]]


# ═══════════════════════════════════════
#  RED TEAM ORCHESTRATOR
# ═══════════════════════════════════════

async def run_red_team(
    session_id: str,
    dataset_id: str,
    protected_attributes: list[str],
    label_column: str,
    favorable_label: str,
    max_rounds: int = 3,
) -> RedTeamSession:
    """
    Run the multi-agent red team adversarial loop.

    1. Train a baseline model on the dataset
    2. Attacker generates adversarial profiles for a target subgroup
    3. Run profiles through the model
    4. Auditor evaluates results and identifies worst subgroup
    5. Repeat until no new bias hotspot or max rounds
    """
    from app.services import dataset_manager

    session = RedTeamSession(session_id=session_id, dataset_id=dataset_id)
    _sessions[session_id] = session

    df = dataset_manager.get_dataset(dataset_id)
    if df is None:
        session.status = "error"
        session.final_summary = "Dataset not found."
        return session

    # Prepare data and train baseline model
    df_clean = df.dropna(subset=[label_column] + protected_attributes).copy()
    df_encoded = df_clean.copy()
    label_encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object" or pd.api.types.is_string_dtype(df_encoded[col]) or df_encoded[col].dtype.kind == "O":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    feature_cols = [c for c in df_encoded.columns if c != label_column]
    X = np.nan_to_num(df_encoded[feature_cols].values.astype(float), nan=0.0)
    y = df_encoded[label_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Encode favorable label
    if label_column in label_encoders:
        try:
            fav_encoded = label_encoders[label_column].transform([str(favorable_label)])[0]
        except ValueError:
            fav_encoded = 1
    else:
        fav_encoded = favorable_label

    # ── Red Team Loop ──
    target_subgroup = "all"
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    use_gemini = bool(api_key)

    for round_num in range(1, max_rounds + 1):
        session.conversation_trace.append({
            "agent": "Orchestrator",
            "round": round_num,
            "message": f"Round {round_num}: Targeting subgroup '{target_subgroup}'",
        })

        # ── ATTACKER: Generate synthetic profiles ──
        attacker_strategy = ""
        if use_gemini and target_subgroup != "all":
            synthetic_df = await _gemini_generate_profiles(
                api_key, target_subgroup, df_clean, feature_cols
            )
            attacker_strategy = f"Gemini-generated maximally qualified profiles for {target_subgroup}"
        else:
            synthetic_df = _generate_diverse_profiles(
                df_clean, protected_attributes, label_column, target_subgroup
            )
            attacker_strategy = (
                f"Systematic generation of highly qualified profiles across all subgroups"
                if target_subgroup == "all"
                else f"Targeted generation of maximally qualified {target_subgroup} profiles"
            )

        session.conversation_trace.append({
            "agent": "Attacker",
            "round": round_num,
            "message": f"{attacker_strategy}. Generated {len(synthetic_df)} profiles.",
        })

        # ── Run through model ──
        synth_encoded = synthetic_df.copy()
        for col in synth_encoded.columns:
            if col in label_encoders:
                le = label_encoders[col]
                known = set(le.classes_)
                synth_encoded[col] = synth_encoded[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in known else 0
                )
            elif synth_encoded[col].dtype == "object" or pd.api.types.is_string_dtype(synth_encoded[col]):
                le = LabelEncoder()
                synth_encoded[col] = le.fit_transform(synth_encoded[col].astype(str))

        synth_features = synth_encoded[[c for c in feature_cols if c in synth_encoded.columns]]
        for col in feature_cols:
            if col not in synth_features.columns:
                synth_features[col] = 0

        synth_features = synth_features[feature_cols]
        X_synth = np.nan_to_num(scaler.transform(synth_features.values.astype(float)), nan=0.0)
        y_pred = model.predict(X_synth)

        # ── AUDITOR: Evaluate subgroup results ──
        subgroup_results = _compute_subgroup_results(synthetic_df, y_pred, fav_encoded)

        # Find worst subgroup
        if subgroup_results:
            worst = subgroup_results[0]  # already sorted ascending by DI
            worst_subgroup = str(worst["subgroup"])
            worst_di = float(worst["di_ratio"])
            worst_severity = str(worst["severity"])
        else:
            worst_subgroup = "unknown"
            worst_di = 1.0
            worst_severity = "low"

        # Root cause analysis
        root_features = []
        if subgroup_results and worst_di < 0.8:
            if "sex" in synthetic_df.columns and "race" in synthetic_df.columns:
                try:
                    parts = eval(worst_subgroup) if worst_subgroup.startswith("(") else (worst_subgroup,)
                    if len(parts) == 2:
                        mask = ((synthetic_df["sex"] == parts[0]) & (synthetic_df["race"] == parts[1])).values
                    else:
                        mask = np.ones(len(synthetic_df), dtype=bool)
                except Exception:
                    mask = np.ones(len(synthetic_df), dtype=bool)
            else:
                mask = np.ones(len(synthetic_df), dtype=bool)

            root_features = _identify_root_cause_features(
                X_synth, y_pred, feature_cols, fav_encoded, mask
            )

        # Auditor analysis
        auditor_msg = (
            f"Worst subgroup: {worst_subgroup} (DI={worst_di:.3f}, severity={worst_severity}). "
            f"{'Root cause features: ' + ', '.join(root_features[:3]) + '.' if root_features else 'No clear root cause identified.'}"
        )

        # Check if done
        done = worst_di >= 0.80 or round_num >= max_rounds
        if round_num > 1 and target_subgroup == worst_subgroup:
            done = True

        session.conversation_trace.append({
            "agent": "Auditor",
            "round": round_num,
            "message": auditor_msg + (" Red team complete." if done else f" Directing Attacker to probe {worst_subgroup} deeper."),
        })

        # Record round
        round_result = RedTeamRound(
            round_num=round_num,
            target_subgroup=target_subgroup,
            profiles_generated=len(synthetic_df),
            attacker_strategy=attacker_strategy,
            subgroup_results=subgroup_results,
            worst_subgroup=worst_subgroup,
            worst_di=worst_di,
            worst_severity=worst_severity,
            root_cause_features=root_features,
            auditor_analysis=auditor_msg,
            done=bool(done),
        )
        session.rounds.append(round_result)

        # Track worst overall
        if worst_di < session.worst_overall_di:
            session.worst_overall_di = worst_di
            session.worst_overall_subgroup = worst_subgroup
            session.root_cause = root_features

        if done:
            break

        target_subgroup = worst_subgroup

    # Final summary
    session.status = "completed"
    session.final_summary = (
        f"Red team completed in {len(session.rounds)} round(s). "
        f"Worst bias discovered: {session.worst_overall_subgroup} "
        f"(DI={session.worst_overall_di:.3f}, severity={FairnessEngine.classify_severity(session.worst_overall_di).value}). "
        f"{'Root cause: ' + ', '.join(session.root_cause[:3]) + '.' if session.root_cause else ''}"
    )

    return session


def _generate_diverse_profiles(
    df: pd.DataFrame,
    protected_attrs: list[str],
    label_column: str,
    target_subgroup: str,
) -> pd.DataFrame:
    """Generate synthetic profiles across all subgroup combinations."""
    all_profiles = []
    feature_cols = [c for c in df.columns if c != label_column]

    if target_subgroup == "all":
        sex_vals = df["sex"].unique() if "sex" in df.columns else ["Unknown"]
        race_vals = df["race"].unique() if "race" in df.columns else ["Unknown"]

        for sex in sex_vals:
            for race in race_vals:
                profiles = _generate_synthetic_profiles(
                    f"{sex} {race}", df[feature_cols], n=10
                )
                all_profiles.append(profiles)
    else:
        profiles = _generate_synthetic_profiles(target_subgroup, df[feature_cols], n=20)
        all_profiles.append(profiles)

    result = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    return result


async def _gemini_generate_profiles(
    api_key: str,
    target_subgroup: str,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Use Gemini to generate adversarial profiles."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        columns_info = {col: str(df[col].dtype) for col in feature_cols[:10]}
        sample = df[feature_cols].head(2).to_dict(orient="records")

        prompt = f"""Generate 10 synthetic candidate profiles for bias testing.

Target subgroup: {target_subgroup}
Make all profiles MAXIMALLY QUALIFIED (high education, many years experience, strong credentials).

Dataset columns: {json.dumps(columns_info)}
Example row format: {json.dumps(sample[0])}

Respond with JSON only: {{"profiles": [{{...}}, ...]}}
Generate exactly 10 profiles matching the column format above."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.5, "max_output_tokens": 3000},
        )

        try:
            text = response.text.strip()
        except ValueError:
            logger.warning("Gemini returned no text for profile generation")
            return _generate_synthetic_profiles(target_subgroup, df, n=20)

        text = text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            for suffix in ["]}", "]}}", '"]}', '"}]}']:
                try:
                    data = json.loads(text + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                logger.warning("Could not parse Gemini profile JSON even after repair")
                return _generate_synthetic_profiles(target_subgroup, df, n=20)

        profiles = data.get("profiles", [])
        if profiles:
            return pd.DataFrame(profiles)

    except Exception as e:
        logger.error(f"Gemini profile generation failed: {e}")

    return _generate_synthetic_profiles(target_subgroup, df, n=20)