# Merge Conflict? Already Fixed. Your AI Just Beat You to It.

*`pr-review-env`: an RL-trained agent that automatically triages pull requests, resolves merge conflicts, and fixes issues — before a human even opens the diff.*

---

> Pull request queues are where engineering velocity goes to die.

The average developer spends **4–6 hours per week** buried in PR backlogs. Merge conflicts pile up and block deploys. Security vulnerabilities sit undetected for days. A single missed null-check or off-by-one bug ships to production and takes down a service at 2 AM.

The bottleneck isn't talent — it's bandwidth.

We built `pr-review-env` to eliminate that bottleneck entirely. It's an RL-trained LLM agent that automatically triages pull requests, resolves merge conflicts, identifies bugs, flags security flaws, and fixes dependency issues — with structured, evidence-grounded explanations a human engineer can actually trust. And unlike most "AI for code" tools, we can **prove it gets better.**

---

## Not a Reviewer. A Resolver.

Most PR automation tools are classifiers. They label things and wait for a human to act. Our agent **resolves** — it reads the pull request, understands what's broken or risky, and produces a complete, actionable fix decision in one shot.

Given a pull request, it runs a **3-stage resolution protocol** — `identify_risk` → `assess_impact` → `final_triage` — and produces four outputs simultaneously:

| Output | What it means |
|---|---|
| **Decision** | `approve`, `request_changes`, or `close` |
| **Risk Assessment** | Category and severity mapped to a structured label set |
| **Priority Judgment** | `low` / `medium` / `high` / `critical` |
| **Resolution Summary** | Evidence-grounded — pulled from the actual diff and file changes |

No vibes-based summarization. No generic "LGTM." Every output is validated against a strict JSON schema — reproducible, structured, and machine-verifiable.

The key insight: **action and explanation must agree.** An agent that flags a PR as critical risk but then approves it isn't resolving anything — it's hallucinating. Our consistency enforcement ensures what the agent *does* and what it *says* are always aligned.

---

## 100 Fixtures. 8 Categories. Every Kind of PR Hell.

We curated **100 real-world PR scenarios** — the kind that actually break things in production — across eight categories, tiered by difficulty:

- **Documentation & Hygiene** — Typos, broken doc links, dead code, renames.
- **Code Quality & Correctness** — Off-by-one bugs, null handling, crash fixes. The 3 AM pages category.
- **Dependency & Config Safety** — Vulnerable pinning, deprecated settings. The "why is this breaking in prod?" category.
- **Security Issues** — SQL injection, XSS, JWT flaws, hardcoded credentials, wildcard CORS.
- **Data & API Integrity** — Schema migrations, breaking API changes, webhook reliability.
- **Performance & Reliability** — Thundering herd, caching pitfalls, connection pooling. Bugs that only appear at 10× load.
- **Concurrency & Distributed Systems** — Race conditions, TOCTOU, deadlocks, distributed locks. The one where even senior engineers get it wrong.
- **Platform & Infrastructure** — Kubernetes probe behavior. One line that takes down an entire cluster.

| Tier | Count | Examples |
|---|---|---|
| 🟢 **Easy** | 30 | off-by-one fix, import sorting, dependency pinning |
| 🟡 **Medium** | 35 | auth middleware, raw SQL endpoint, Redis caching, webhook error handling |
| 🔴 **Hard** | 35 | distributed locking, JWT rotation, queue consumer races, connection pool exhaustion |

---

## Trained to Improve. Wired to Be Fast.

Getting an LLM to resolve a PR once is a prompting problem. Getting it to *improve* is a training problem. That's the part nobody talks about.

We built a full RL pipeline using **GRPO** with **LoRA fine-tuning** via TRL and PEFT. Here's what makes our reward function genuinely different:

- **Four-axis scoring** — decision accuracy, label quality (F1), priority calibration (ordinal distance), and summary relevance. Every axis matters, every step.
- **Stage-aware weights** — during `identify_risk`, summary quality dominates (reasoning first). By `final_triage`, the decision itself takes over. The agent is forced to think, not just guess.
- **⚡ Speed as a first-class metric** — an exponential latency discount penalizes slow resolvers. A model that's 80% accurate in 3 seconds outscores one that's 85% accurate in 30. Optimized for production, not benchmarks.
- **🔒 Consistency enforcement** — penalized whenever action contradicts explanation. A `request_changes` paired with "looks good" gets flagged. Eliminates an entire class of hallucinating outputs.
- **Deterministic grading** — no hidden LLM critic. Every score is computed from visible logic, fully reproducible, fully auditable.

```
adjusted_score = raw_reward × exp(−0.35 × latency_overrun)
```

---

## The Numbers

After **2 epochs of GRPO training** on our 100-fixture dataset, here's what the curves actually show:

### Training Loss — Consistent Descent

Loss dropped steadily from **~1.35 → ~0.28** across 260 steps. No plateaus, no divergence — clean, stable learning throughout.

![GRPO Training Loss Curve](https://raw.githubusercontent.com/mithilesh11705/Meta_3_1/main/training_results/training_loss.jpeg)

### Reward Signal — Upward Trend

Mean reward climbed from **~0.10–0.25** in early steps to consistently hitting **0.55–0.70** by step 240. The ceiling is rising as the model learns the reward structure.

![GRPO Training Reward Curve](https://raw.githubusercontent.com/mithilesh11705/Meta_3_1/main/training_results/reward_curve.png)

### Environment Reward — Model Getting Smarter

The env reward mean (scored directly by the environment, not the trainer) rose from **~0.21 → ~0.42–0.49** over 2 epochs. This is the number that matters — it's the agent getting genuinely better at resolving real PR scenarios.

![GRPO Env Reward Mean](https://raw.githubusercontent.com/mithilesh11705/Meta_3_1/main/training_results/trainer_env_reward_curve.png)

### Learning Rate & Gradient Norm — Healthy Training Signal

Linear decay from **2e-5 → 0** over 2 epochs with gradient norms oscillating in the **1.5–5.2** range — active, stable optimization with no collapse.

![GRPO Learning Rate](https://raw.githubusercontent.com/mithilesh11705/Meta_3_1/main/training_results/trainer_learning_rate_curve.png) ![GRPO Gradient Norm](https://raw.githubusercontent.com/mithilesh11705/Meta_3_1/main/training_results/trainer_grad_norm_curve.png)

---

### Outcome Summary

| Metric | Result |
|---|---|
| Training Loss | 1.35 → **0.28** (−79%) |
| Env Reward Mean | 0.21 → **0.47** (+124%) |
| Peak Training Reward | **0.70** |
| Test Suite | **398 cases, all green** |
| Resolution Speed | **~3–8 seconds** per PR |

Our test suite covers **398 cases** across reward logic, environment behavior, schema validation, and training utilities — all green. Every score is reproducible. Spin up the [live HF Space](https://huggingface.co/spaces/hitanshjain1812/meta_final/main), run the environment yourself, and verify any claim we make here.

The average engineering team merges **50–200 PRs per week.** If this agent resolves even 40% of them autonomously, that's **20–80 hours of engineering time back per week.**

That's not a benchmark number. That's a business case.

---

*398 tests · 100 fixtures · 8 problem categories · One agent that resolves.*
*`pr-review-env` — built in a hackathon, designed for production.*
