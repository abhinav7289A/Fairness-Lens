<p align="center">
  <img src="https://img.shields.io/badge/Google%20Solution%20Challenge-2026-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Theme-Unbiased%20AI%20Decision-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-10B981?style=for-the-badge" />
</p>

<h1 align="center">⚖️ FairnessLens</h1>
<h3 align="center">AI Bias Detection, Mitigation & Deployment Validation Platform</h3>

<p align="center">
  <strong>Inspect → Measure → Flag → Fix → Validate</strong><br/>
  A five-phase pipeline that audits ML models for discriminatory behavior,<br/>
  fixes the bias, proves the fix works, and generates regulatory-compliant reports.
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-pipeline-deep-dive">Pipeline Deep Dive</a> •
  <a href="#-advanced-features">Advanced Features</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

---

## 🎯 Problem Statement

Companies increasingly use AI to screen job applications, approve loans, and assess criminal risk. These algorithms can quietly discriminate — rejecting qualified candidates based on gender, race, or age — while appearing objective. The people harmed never know why they were rejected, and the compliance officers responsible for fairness lack the technical tools to audit these systems.

**FairnessLens** opens the black box. It measures bias with eight mathematically rigorous metrics, fixes it with four proven techniques, validates the fix with three deployment-readiness tests, and generates audit reports aligned with real regulations (NYC Local Law 144, EEOC Four-Fifths Rule, EU AI Act).

---

## 🌐 Live Demo

| Component | URL |
|-----------|-----|
| **Frontend** | [fairness-lens.vercel.app](https://fairness-lens.vercel.app) |
| **Backend API** | [fairness-lens-32190750126.us-central1.run.app](https://fairness-lens-32190750126.us-central1.run.app) |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Vercel)                         │
│              Next.js 14 · Tailwind · Recharts               │
│                                                             │
│  ┌──────┐ ┌───────┐ ┌──────┐ ┌─────┐ ┌──────┐ ┌────────┐  │
│  │Upload│→│Inspect│→│Measure│→│Flag │→│ Fix  │→│Validate│  │
│  └──────┘ └───────┘ └──────┘ └─────┘ └──────┘ └────────┘  │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────┐  │
│  │AI Agent  │ │Red Team  │ │Counterfactual│ │RL Optimizer│  │
│  └──────────┘ └──────────┘ └──────────────┘ └───────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────┴────────────────────────────────────┐
│                 BACKEND (Google Cloud Run)                   │
│                    FastAPI · Python 3.11                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              17 API Endpoints (11 Routers)          │    │
│  │  inspect · measure · flag · fix · validate          │    │
│  │  report · model · agent · redteam                   │    │
│  │  counterfactual · rl_fix                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  AIF360 +    │  │ NVIDIA Gemma │  │ Gemini 2.5 Flash │  │
│  │  Fairlearn   │  │  3 27B (LLM) │  │ (Function Call)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**LLM Routing Strategy:**

| Function | Provider | Model | Why |
|----------|----------|-------|-----|
| `explain_bias()` | NVIDIA | Gemma 3 27B | High-volume, no function calling needed |
| `explain_mitigation()` | NVIDIA | Gemma 3 27B | High-volume, plain text output |
| `generate_audit_narrative()` | NVIDIA | Gemma 3 27B | Called per PDF download |
| AI Audit Agent | Google | Gemini 2.5 Flash | Requires native function calling |
| Red Team Attacker/Auditor | Google | Gemini 2.5 Flash | Requires structured JSON + tool use |
| Counterfactual Explainer | — | Pure algorithm | DiCE-inspired greedy search |
| RL Optimizer | — | Pure algorithm | Numpy-only DQN |

---

## 🔬 Pipeline Deep Dive

### Phase 1: Inspect — Dataset Profiling

The Inspect phase answers: *what does this data look like, and where might bias hide?*

**Five analyses, zero LLM calls:**

**1. Demographic Detection**
Scans column names against a dictionary of protected attribute terms (`gender`, `sex`, `race`, `ethnicity`, `age`, `disability`, `religion`, `national_origin`) using case-insensitive substring matching.

**2. Distribution Analysis**
For each protected group, computes value counts, proportions, and favorable outcome rates:

```
rate(g) = count(favorable_outcome ∧ group=g) / count(group=g)
```

**3. Proxy Variable Detection**
Catches indirect discrimination (e.g., zip code correlating with race). For each non-protected feature, computes correlation with each protected attribute. A feature is flagged as a proxy if `|r| > 0.3`.

- **Numeric vs Binary:** Point-biserial correlation
- **Categorical vs Categorical:** Cramér's V

```
Cramér's V = √( χ² / (n × min(r-1, c-1)) )

where χ² = chi-squared statistic on the contingency table
      n   = total observations
      r,c = rows and columns of the contingency table
```

**4. Representation Gap Analysis**
Compares dataset demographics to U.S. Census baselines:

```
Gap = dataset_percentage − census_percentage
```

A +37.2% gap for White means the model has limited exposure to other groups.

**5. Warning System**
Auto-generates warnings for: small group sizes (< 30 samples), extreme imbalances (< 5% representation), high missing-data rates (> 10% nulls), and detected proxy variables.

---

### Phase 2: Measure — Eight Fairness Metrics

This is the analytical core. Each metric captures a different mathematical definition of "fairness." The **Chouldechova 2017 impossibility theorem** proves that when base rates differ between groups, calibration, predictive parity, and equalized odds cannot all hold simultaneously — so we measure all eight and let the user see the trade-offs.

#### Metric 1: Statistical Parity Difference (SPD)

```
SPD = P(Ŷ=1 | A=unprivileged) − P(Ŷ=1 | A=privileged)
```

Measures the gap in favorable-outcome rates between groups.
- **Range:** [-1, 1]
- **Ideal:** 0
- **Threshold:** |SPD| ≤ 0.1
- **Use when:** Historical data itself is biased (hiring from a male-dominated industry)

#### Metric 2: Disparate Impact Ratio (DI) — The Four-Fifths Rule

```
DI = rate(unprivileged) / rate(privileged)
```

The EEOC's legal standard. If DI < 0.80, the model creates a *prima facie* case of adverse impact under Title VII.
- **Range:** [0, ∞)
- **Ideal:** 1.0
- **Legal threshold:** DI ≥ 0.80

```
┌────────────┬───────────────┬──────────────────────────┐
│ DI Range   │ Severity      │ Action Required          │
├────────────┼───────────────┼──────────────────────────┤
│ ≥ 0.90     │ Low (Green)   │ Monitor and document     │
│ 0.80–0.90  │ Medium (Yellow)│ Investigate causes      │
│ 0.65–0.80  │ High (Orange) │ Immediate mitigation     │
│ < 0.65     │ Critical (Red)│ Halt deployment          │
└────────────┴───────────────┴──────────────────────────┘
```

#### Metric 3: Average Absolute Odds Difference (AOD) — Equalized Odds

```
AOD = 0.5 × ( |TPR_unpriv − TPR_priv| + |FPR_unpriv − FPR_priv| )
```

Both true positive and false positive rates should be equal across groups.
- **Range:** [0, 1]
- **Ideal:** 0
- **Threshold:** AOD ≤ 0.1
- **Use when:** Both false positives and false negatives cause harm (criminal justice, medical diagnosis)

#### Metric 4: Equal Opportunity Difference (EOP)

```
EOP = P(Ŷ=1|Y=1,A=unpriv) − P(Ŷ=1|Y=1,A=priv)
```

Given someone *should* be selected, are they? Focuses only on qualified candidates.
- **Range:** [-1, 1]
- **Ideal:** 0
- **Threshold:** |EOP| ≤ 0.1
- **Use when:** You want to ensure qualified minority candidates aren't missed

#### Metric 5: Predictive Parity Difference (PPD)

```
PPD = PPV_unpriv − PPV_priv

where PPV = TP / (TP + FP)
```

When the model predicts positive, how often is it right — for each group?
- **Threshold:** |PPD| ≤ 0.1
- **Use when:** Risk scores should mean the same thing across groups

#### Metric 6: Calibration Difference

Bins probabilistic predictions into deciles. For each bin, computes actual positive rate per group. Averages absolute differences across bins.
- **Threshold:** ≤ 0.1
- **Use when:** Credit scoring, probability-based decisions

#### Metric 7: Individual Fairness (k-NN Consistency)

```
Consistency = (1/n) Σᵢ I(Ŷᵢ = mode({Ŷⱼ : j ∈ kNN(i)}))
```

For each instance, checks whether its prediction matches the majority prediction among its k=5 nearest neighbors. Similar individuals should be treated similarly.
- **Threshold:** ≥ 0.7

#### Metric 8: Counterfactual Fairness

For each instance, changes only the protected attribute and re-predicts. If the prediction flips, the model is counterfactually unfair.
- **Flip rate threshold:** ≤ 0.05
- Directly tests whether the protected attribute *itself* drives decisions

#### Intersectional Analysis (Required by NYC LL144)

Computes impact ratios for every `(sex × race)` subgroup combination, comparing each to the highest-performing intersection. Reveals compounding bias — Black women may face worse outcomes than either Black men or White women alone.

---

### Phase 3: Flag — Risk Assessment & Compliance

The Flag phase transforms raw metrics into actionable compliance intelligence. This is the first phase that calls an LLM — **NVIDIA Gemma 3 27B** — for plain-English bias translation.

**Components:**

1. **Bias Scorecard** — Modeled on Google Model Cards (Mitchell 2019) and NYC LL144 audit format
2. **Regulatory Compliance Checks:**

| Regulation | Rule | Penalty |
|------------|------|---------|
| NYC Local Law 144 | Annual bias audits with intersectional impact ratios | $500 first violation, $1,500/day |
| EEOC Four-Fifths Rule | Selection rate ≥ 80% of highest group | Evidence of adverse impact (Title VII) |
| EU AI Act (Aug 2026) | Representative training data, free of errors | Up to €35M or 7% global turnover |

3. **Gemma 3 27B Explanation** — Returns JSON with `summary`, `severity`, `affected_groups`, `plain_english` (2-3 paragraphs), and `recommendations` (3-5 actionable steps). Four-tier JSON repair fallback handles malformed LLM responses.

---

### Phase 4: Fix — Four Mitigation Techniques

Most competing tools stop at detection. FairnessLens actively fixes bias.

#### Technique 1: Reweighting (Pre-processing)

Computes per-sample weights that make the weighted dataset achieve demographic parity:

```
W(group=g, label=l) = P(l) × P(g) / P(g,l)
```

Under-represented `(group, label)` pairs get higher weights. The model trains with `sample_weight=W`.

**Example:** If 80% of selected candidates are male, male-selected pairs get lower weight while female-selected pairs get higher weight, forcing the model to value both equally.

#### Technique 2: Disparate Impact Remover (Pre-processing)

Feinberg 2015 rank-preserving repair. For each non-protected feature:
1. Compute per-group quantile distribution
2. Shift each group's values toward a shared median
3. `repair_level ∈ [0, 1]` controls intensity

Removes correlation with protected attribute while preserving within-group ranking.

#### Technique 3: Exponentiated Gradient (In-processing)

Fairlearn's constrained optimization. Wraps any sklearn estimator and iteratively reweights during training to satisfy fairness constraints (`DemographicParity` or `EqualizedOdds`).

```
min  loss(θ)
s.t. fairness_violation(θ) ≤ ε
```

Provides the best fairness-accuracy balance when you have training access.

#### Technique 4: Threshold Optimizer (Post-processing)

Finds per-group classification thresholds without retraining:

```
For each group g, find τ_g such that:
P(score ≥ τ_g | A=g) is equal across all groups
```

Use when you have black-box access only (can't retrain, only adjust decision boundaries).

#### Recommendation Engine

Automatically ranks techniques by a weighted score:

```
score = fairness_improvement × 0.7 + (1 − accuracy_cost) × 0.3
```

The technique with the highest score is recommended with a gold border in the UI.

---

### Phase 5: Validate — Deployment Readiness

After Fix, how do you know the mitigated model works in production? Three industry-standard tests produce a **Deployment Readiness Score** out of 100.

#### Test 1: Fresh Cohort Simulation (40 points)

Generates **500 synthetic candidates** from a deliberately shifted distribution — minority groups get 60% representation instead of the dataset's actual breakdown. Runs both original and mitigated models on this fresh cohort.

```
┌─────────────────────┬────────┬───────────┐
│ DI on Fresh Cohort  │ Points │ Verdict   │
├─────────────────────┼────────┼───────────┤
│ ≥ 0.80              │ 40     │ generalizes│
│ 0.70 – 0.80         │ 30     │ acceptable│
│ 0.60 – 0.70         │ 20     │ concerning│
│ 0.50 – 0.60         │ 10     │ overfitted│
│ < 0.50              │ 0      │ brittle   │
└─────────────────────┴────────┴───────────┘
```

#### Test 2: Shadow Deployment Disagreement (35 points)

Runs original and mitigated models on the same test set. For every candidate where they disagree:

```
favorable_flip_ratio = flips_favoring_unprivileged / total_flips
```

A fair mitigation should produce disagreements that move decisions toward previously-disadvantaged groups.

```
┌──────────────────────┬────────┬──────────────────┐
│ Favorable Flip Ratio │ Points │ Verdict          │
├──────────────────────┼────────┼──────────────────┤
│ ≥ 0.80               │ 35     │ strong_correction│
│ 0.60 – 0.80          │ 25     │ partial_correction│
│ 0.40 – 0.60          │ 15     │ mixed            │
│ < 0.40               │ 5      │ harmful          │
└──────────────────────┴────────┴──────────────────┘
```

#### Test 3: Stability Under Perturbation (25 points)

Selects 20 candidates. For each, generates 50 variants with small Gaussian noise (σ=0.05 on scaled features):

```
consistency(x) = (1/50) Σᵢ I(predict(x + εᵢ) = predict(x))
```

A robust model scores ≥ 95% consistency. Below 80% indicates brittle predictions.

#### Deployment Readiness Badges

```
Total = Fresh Cohort + Shadow + Stability (max 100)

┌───────────┬──────────────────────┬─────────────────────────┐
│ Score     │ Badge                │ Recommendation          │
├───────────┼──────────────────────┼─────────────────────────┤
│ 85 – 100  │ ✅ Ready to Deploy   │ Safe for production     │
│ 70 – 84   │ ⚠️ Deploy w/ Monitor │ Track drift in prod     │
│ 50 – 69   │ 🟠 Needs More Work  │ Apply stronger mitigation│
│ 0 – 49    │ 🔴 Do Not Deploy    │ Block deployment        │
└───────────┴──────────────────────┴─────────────────────────┘
```

#### Three-Way Validation (Original vs Standard vs RL)

The Validate phase can extend to include an RL-discovered mitigation sequence (Reweighting → Threshold Optimizer), producing a side-by-side three-way comparison with stacked gauge cards and two narrative framings:

- **"RL is better"** — Multi-step RL discovers sequences single techniques cannot
- **"Different tools for different constraints"** — Standard for fast deployment, RL when compute allows

---

## 🚀 Advanced Features

### 🤖 AI Audit Agent

An autonomous **Gemini 2.5 Flash** agent that replaces the manual wizard. The user types a single natural-language instruction like *"Audit this model for racial bias"* and the agent plans and executes the full pipeline.

**ReAct (Reason + Act) Loop:**

```
┌──────────┐    ┌──────────┐    ┌─────────────┐
│ THINKING │ →  │  ACTION  │ →  │ OBSERVATION │ ──┐
│ (reason) │    │ (tool)   │    │  (result)   │   │
└──────────┘    └──────────┘    └─────────────┘   │
     ↑                                            │
     └────────────────────────────────────────────┘
                    (max 6 iterations)
```

**5 Tools Available:**

| Tool | Maps To |
|------|---------|
| `profile_dataset` | `DataProfiler.run_full_inspection()` |
| `compute_metrics` | `FairnessEngine.compute_all_metrics()` |
| `flag_issues` | Flag route logic + NVIDIA explanation |
| `apply_mitigation` | MitigationService |
| `generate_report` | PDF report generator |

---

### ⚔️ Multi-Agent Red Team Tester

Two specialized Gemini agents in an adversarial loop to surface hidden bias that standard metrics might miss.

```
┌──────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                       │
│              Manages rounds and state                 │
└───────────┬──────────────────────────┬───────────────┘
            │                          │
   ┌────────▼────────┐       ┌────────▼────────┐
   │    ATTACKER      │       │     AUDITOR      │
   │                  │       │                  │
   │ Generates edge-  │       │ Evaluates model  │
   │ case profiles:   │  ───► │ predictions:     │
   │ maximally        │       │ • Selection rates│
   │ qualified from   │       │ • DI per subgroup│
   │ underrepresented │       │ • Root cause     │
   │ subgroups        │       │ • Next target    │
   └──────────────────┘       └──────────────────┘
```

**Termination Conditions:**
- Worst DI exceeds 0.80 (bias resolved)
- Same subgroup probed twice with no new finding
- `max_rounds` (default 3) reached

**Output:** Worst discovered subgroup with root-cause feature attribution and severity rating.

---

### 🔀 Counterfactual Fairness Explainer

For each rejected candidate from the unprivileged group, finds the **minimal feature change** that would flip the prediction — while keeping protected attributes **locked**.

**DiCE-Inspired Greedy Search Algorithm:**

```
1. Get model's probability for original profile
2. For each non-protected feature:
   - Try perturbing to values that push probability toward favorable
3. Greedily pick the change with highest probability gain
4. Apply it, re-predict
5. Repeat until prediction flips or max_changes=4 reached
6. Generate plain-English narrative
```

**Example Output:**
> *"Priya Sharma was rejected. Changing just 2 features (education_num from 13 → 18.2, age from 56 → 81.9) — while keeping sex=Female, race=Black unchanged — would flip the decision to Selected. These features act as proxies for the protected attributes."*

Any flip reveals **proxy discrimination**, not direct discrimination — the protected attributes were locked, so other features are doing the discriminating.

---

### 🧠 Reinforcement Learning Mitigation Optimizer

Frames bias mitigation as a **Markov Decision Process**. A Deep Q-Network (numpy-only, no PyTorch) learns which *sequence* of mitigation actions maximizes fairness while minimizing accuracy loss.

#### MDP Formulation

```
State (7-dim):  [DI_ratio, |SPD|, EOD, |EOP|, |PPD|, accuracy, step/max_steps]
Actions (9):    [reweight, threshold_DP, threshold_EO, DIR_low, DIR_high,
                 reweight→threshold, DIR→reweight, DIR→threshold, STOP]
Reward:         R = 3×Δfairness − λ×Δaccuracy_loss + 2.0 (if DI crosses 0.80)
```

#### DQN Architecture

```
Input (7) → Dense(64, ReLU) → Dense(32, ReLU) → Output (9 Q-values)

Training:
  • Replay buffer: 3000 experiences
  • Batch size: 32
  • Epsilon: 1.0 → 0.05 (decay rate 0.96)
  • Target network sync: every 8 episodes
  • Episodes: 80, max steps per episode: 5
```

#### Brute-Force Floor Guarantee

Before DQN training, evaluates 10 hand-crafted seed sequences and picks the best as a **floor**. The DQN result must match or beat this floor — guaranteeing the RL optimizer **never does worse** than simple techniques.

#### Pareto Frontier

For seven λ values in `[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]`, finds the best sequence within a λ-appropriate candidate pool. Plots accuracy vs DI ratio so users can choose their trade-off:

```
Accuracy ↑
    │      ★ λ=2.0 (accuracy-first)
    │    ★ λ=1.0
    │  ★ λ=0.5
    │ ★ λ=0.3
    │★ λ=0.1
    │★ λ=0.0 (fairness-first)
    └──────────────────────→ DI Ratio
```

---

## 📄 PDF Audit Report

Generates a five-section compliance-ready PDF via ReportLab, suitable for regulatory filing:

1. **Title Page** — Dataset name, date, row/column count, overall severity
2. **Inspect** — Profile, distributions, proxies, representation gaps, warnings
3. **Measure** — All 8 metrics with values, thresholds, pass/fail
4. **Flag** — Severity, compliance checks, flagged issues, LLM explanations
5. **Fix** — Technique comparison, before/after metrics, recommendation
6. **Validate** — Deployment readiness scores, three-way comparison (if RL run)

---

## 📊 Demo Datasets

Three pre-loaded datasets for immediate testing:

| Dataset | Rows | Task | Protected Attributes | Known Bias |
|---------|------|------|---------------------|------------|
| **Adult Income** | 32,561 | Income >$50K prediction | Sex, Race | Women selected at 36% the rate of men |
| **German Credit** | 1,000 | Credit risk classification | Sex, Age | Young applicants penalized |
| **COMPAS** | 7,214 | Recidivism prediction | Sex, Race | ProPublica: biased against Black defendants |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (backend)
- **Node.js 18+** (frontend)
- **NVIDIA API Key** — free at [build.nvidia.com](https://build.nvidia.com) (for bias explanations)
- **Google Gemini API Key** — free at [aistudio.google.com](https://aistudio.google.com/apikey) (for AI Agent & Red Team)

### Local Development Setup

**1. Clone the repository**

```bash
git clone https://github.com/abhinav7289A/Fairness-Lens.git
cd Fairness-Lens
```

**2. Backend setup**

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
# Add your actual API keys (no quotes, no trailing spaces)
echo GOOGLE_API_KEY=your_gemini_key_here > .env
echo NVIDIA_API_KEY=your_nvidia_key_here >> .env

# Start the backend server
uvicorn app.main:app --reload --port 8000
```

The backend will be running at `http://localhost:8000`. Verify by visiting `http://localhost:8000/health`.

**3. Frontend setup (new terminal)**

```bash
cd frontend

# Install dependencies
npm install

# Create environment file pointing to local backend
echo NEXT_PUBLIC_API_URL=http://localhost:8000 > .env.local

# Start the development server
npm run dev
```

The frontend will be running at `http://localhost:3000`.

**4. Open in browser**

Navigate to `http://localhost:3000`. Click "Try Demo" on any dataset to start.

### Production Deployment

**Backend — Google Cloud Run:**

```bash
# Push to GitHub (Cloud Build trigger auto-deploys)
git push origin main

# Or manual deploy
gcloud run deploy fairness-lens \
  --source . \
  --region us-central1 \
  --set-env-vars GOOGLE_API_KEY=xxx,NVIDIA_API_KEY=xxx,CORS_ORIGINS=https://your-app.vercel.app
```

**Frontend — Vercel:**

1. Import repo at [vercel.com](https://vercel.com)
2. Set **Root Directory** to `frontend`
3. Add env var: `NEXT_PUBLIC_API_URL` = your Cloud Run URL
4. Deploy — auto-redeploys on every push

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend Framework** | FastAPI | 17 API endpoints, async support |
| **Fairness Libraries** | AIF360 + Fairlearn | Industry-standard bias metrics and mitigation |
| **ML** | scikit-learn | Model training, preprocessing |
| **Manual Pipeline LLM** | NVIDIA Gemma 3 27B | Bias explanations, mitigation text, audit narratives |
| **Advanced Features LLM** | Google Gemini 2.5 Flash | Function calling for AI Agent and Red Team |
| **PDF Generation** | ReportLab | Compliance-ready audit reports |
| **Frontend Framework** | Next.js 14 | React server components, app router |
| **Styling** | Tailwind CSS | Utility-first responsive design |
| **Charts** | Recharts | Radar charts, bar charts, line charts |
| **Backend Hosting** | Google Cloud Run | Auto-scaling, pay-per-use |
| **Frontend Hosting** | Vercel | Edge network, auto-deploy |
| **CI/CD** | GitHub Actions + Cloud Build | Import checks, auto-deploy on push |

---

## 📁 Project Structure

```
fairness-lens/                        (65 files)
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI entry, 11 routers, CORS
│   │   ├── core/
│   │   │   ├── fairness.py           # 8 fairness metrics engine
│   │   │   ├── gemini.py             # NVIDIA + Gemini LLM client
│   │   │   └── utils.py              # Pandas 3.x compatibility helpers
│   │   ├── api/routes/               # 11 route files
│   │   │   ├── inspect.py            # Dataset profiling
│   │   │   ├── measure.py            # Fairness metrics
│   │   │   ├── flag.py               # Risk assessment
│   │   │   ├── fix.py                # Mitigation
│   │   │   ├── validate.py           # Deployment validation
│   │   │   ├── report.py             # PDF generation
│   │   │   ├── model.py              # Model upload
│   │   │   ├── agent.py              # AI Audit Agent
│   │   │   ├── redteam.py            # Red Team testing
│   │   │   ├── counterfactual.py     # Counterfactual explainer
│   │   │   └── rl_fix.py             # RL optimizer
│   │   └── services/                 # Business logic
│   │       ├── dataset_manager.py    # In-memory dataset storage
│   │       ├── mitigation.py         # 4 mitigation techniques
│   │       ├── validate.py           # 3 deployment tests
│   │       ├── agent.py              # ReAct agent orchestrator
│   │       ├── redteam.py            # Multi-agent red team
│   │       ├── counterfactual.py     # DiCE-inspired explainer
│   │       ├── rl_optimizer.py       # DQN mitigation optimizer
│   │       └── pdf_report.py         # ReportLab PDF builder
│   ├── requirements.txt
│   ├── Dockerfile
│   └── runtime.txt
├── frontend/
│   ├── src/
│   │   ├── app/page.js               # Main orchestrator (all state)
│   │   ├── components/
│   │   │   ├── Header.jsx            # Top bar, dark mode
│   │   │   ├── Sidebar.jsx           # 6-step navigation
│   │   │   └── steps/
│   │   │       ├── UploadStep.jsx    # Demo/CSV/Model upload
│   │   │       ├── InspectStep.jsx   # Data profiling view
│   │   │       ├── MeasureStep.jsx   # Metrics + radar chart
│   │   │       ├── FlagStep.jsx      # Risk cards + compliance
│   │   │       ├── FixStep.jsx       # Before/after comparison
│   │   │       ├── RLFixStep.jsx     # RL results + Pareto chart
│   │   │       ├── ValidateStep.jsx  # Stacked gauges + 3-way comparison
│   │   │       ├── AgentPanel.jsx    # AI Agent interface
│   │   │       ├── RedTeamPanel.jsx  # Red Team conversation view
│   │   │       └── CounterfactualPanel.jsx
│   │   └── lib/api.js                # API client + demo metadata
│   ├── package.json
│   └── vercel.json
├── .github/workflows/ci.yml
└── README.md
```

---

## 📚 Academic References

| Paper | Year | Used For |
|-------|------|----------|
| Feldman et al. — Certifying and Removing Disparate Impact | 2015 | Disparate Impact Remover |
| Chouldechova — Fair Prediction with Disparate Impact | 2017 | Impossibility theorem explanation |
| Mitchell et al. — Model Cards for Model Reporting | 2019 | Bias scorecard design |
| Kearns et al. — Preventing Fairness Gerrymandering | 2018 | Intersectional fairness approach |
| Rodolfa et al. — Empirical Observation of Negligible Trade-offs | 2021 | Accuracy-fairness trade-off messaging |

---

## 🏆 Google Solution Challenge 2026

**Team Size:** 2
**Theme:** Unbiased AI Decision
**Timeline:** 4 weeks

**Key Differentiators:**
1. Goes beyond detection — actively fixes bias and proves the fix works
2. Regulatory-compliant (NYC LL144, EEOC, EU AI Act) with downloadable audit reports
3. Four advanced AI features: autonomous agent, adversarial red team, counterfactual stories, RL optimization
4. Plain-English explanations for non-technical stakeholders
5. Three-way deployment validation (Original vs Standard vs RL)

---

<p align="center">
  Built with ❤️ for fairness in AI
</p>
