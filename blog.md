---
title: "The Agentic Triage Protocol: Training Professional Code Reviewers with GRPO"
thumbnail: /blog/assets/pr_review_env/thumbnail.png
authors:
- name: Meta Infra Tooling
---

# The Agentic Triage Protocol: Training Professional Code Reviewers with GRPO

In the pressurized world of modern CI/CD, the gap between "code generation" and "code governance" is widening. While LLMs have become proficient at writing code, they remain dangerously naive at **guarding** it. Most models can summarize a diff, but they fail to comprehend the catastrophic potential of a subtle race condition or a missing security boundary.

Today, we are diving deep into the **`pr-review-env`** project—a paradigm shift in how we train and evaluate AI for professional software engineering. By combining a deterministic, high-fidelity environment with the reasoning power of **Group Relative Policy Optimization (GRPO)**, we have moved beyond chat-based assistants toward truly autonomous, senior-grade triage agents.

## The Core Thesis: Beyond Shallow Summarization

The central problem in AI-assisted code review is what we call **"The Saliency Gap."** Most models can summarize *what* changed, but fail to comprehend *why it matters* within a professional risk framework. A simple logger change is a non-event; a subtle shift in how a Redis lock is acquired can bring down a global service.

To solve this, we moved beyond text benchmarks and built a **World Model for Professional Triage.**

---

## 1. The Environment Engine: A High-Fidelity Simulator

The **`pr-review-env`** is not a static dataset; it is an interactive, state-aware protocol. We developed a suite of **100 diversified scenarios**, derived from real-world production post-mortems and security disclosures. 

### A Catalog of Professional Complexity
Our tasks aren't just "find the bug." They are cross-functional puzzles:
- **Reliability & Concurrency**: TOCTOU (Time-of-Check to Time-of-Use) races, distributed lock expirations, and N+1 query patterns.
- **The Security Perimeter**: JWT sanitization, deserialization RCE, and cryptographic nonce reuse.
- **Architectural Stewardship**: Circular dependency detection, breaking API contracts, and dependency pinning.

### The Staged Interaction Protocol
Professional triage isn't a one-shot task. Our environment forces the agent to navigate a three-step cognitive journey:
1.  **Identify Risk (Extraction)**: Parsing the diff and reviewer history to find the core issue.
2.  **Assess Impact (Prioritization)**: Categorizing the risk and assigning priority.
3.  **Final Triage (Remediation)**: Providing remediation guidance and a final merge decision.

This staged approach ensures that the model "shows its work" and aligns its reasoning with senior engineering standards at every step.

---

## 2. Technical Deep Dive: The Staged Reward Logic

The "professionalism" of our agent is enforced through a **Multi-Axis Reward Engine** (`reward.py`). We don't just ask "did it get the answer right?" We grade the **process**.

### Sub-Score Breakdown:
- **Decision Accuracy (30%)**: Did the agent correctly `approve`, `request_changes`, or `close`? This is the baseline of utility.
- **Label F1 Score (25%)**: We use an F1-score to reward precise categorization. An agent that correctly identifies `security` but misses `urgent` is partially rewarded, while an agent that "spam-labels" everything is penalized.
- **Priority Alignment (20%)**: Does the agent realize a TOCTOU bug is `Critical` while a typo is `Low`? This maps the model's understanding of business impact.
- **Evidence-Grounded Prose (25%)**: A deterministic **Evidence Oracle** verifies that the agent is citing the specific files and functions modified in the diff.

### The "Anti-Hedging" Penalty
To prevent reward hacking, we implemented a **Consistency Penalty.** If a model identifies a `security` risk but then `approves` the PR, it receives a heavy penalty. This forces the model to develop a coherent internal logic—a vital trait for any production-ready system.

---

## 3. Training with GRPO: Reasoning Without a Critic

The breakthrough in this project came from our use of **Group Relative Policy Optimization (GRPO)**. Unlike traditional reinforcement learning, GRPO eliminates the need for a separate critic model.

### How it Works in the Triage Context:
1.  **Group Sampling**: The model produces a group of 8-16 potential reviews for the same PR.
2.  **Environment Validation**: Each review is sent to our FastAPI validator.
3.  **Relative Learning**: The model learns to favor the reviews that achieved higher scores *relative* to the others in that same group.

This creates a high-pressure "internal discourse" where the model discovers that technical specificity consistently yields higher rewards than generic politeness. Coupling this with **Unsloth 4-bit LoRA**, we achieved 3x faster training times on a single A100 GPU.

---

## 4. Behind the Scenes: The "Hard" Scenarios

Let's look at one of our "Hard" tasks: **The Redis TOCTOU Race.**

In this scenario, a developer submits a PR intended to add rate-limiting. The logic is:
1. `GET` the current request count.
2. If count < limit, `SET` count + 1.

A generalist model sees "correct logic." A senior reviewer (and our fine-tuned agent) sees a **race condition**—in high-concurrency environments, multiple requests can pass the check simultaneously. Our agent learns to identify this, label it as `bug` and `urgent`, and suggest using a Redis Lua script or `INCR` to ensure atomicity.

---

## 5. Early Experiments: Why Qwen 2.5?

During our development phase, we tested several base models. While Llama 3 was strong in general prose, **Qwen 2.5 (specifically the 3B and 7B Instruct variants)** showed an exceptional affinity for structural reasoning—essential for parsing diffs and generating valid JSON actions. 

We found that Qwen’s training on diverse coding tasks allowed it to "speak the language" of the environment much faster, reducing the number of parse-failure penalties during early GRPO epochs.

---

## 6. Performance Benchmarks: The Results

We evaluated the performance across 100 tasks, measuring the "Reasoning Leap" before and after training.

| Difficulty | Baseline Reward | Post-GRPO Reward | Delta |
| :--- | :---: | :---: | :---: |
| **Easy (30 tasks)** | 0.85 | **0.94** | +10.6% |
| **Medium (35 tasks)** | 0.62 | **0.81** | +30.6% |
| **Hard (35 tasks)** | 0.35 | **0.62** | **+77.1%** |

### The Speed Component
We also implemented a **Latency-Adjusted Score.** Automated triage is only useful if it’s faster than a human. post-training, our agents now deliver senior-grade reviews in **under 8 seconds**—a 45% reduction in latency compared to the baseline.

---

## 7. Metrics That Matter: The Evidence Oracle

A professional review is only as good as its grounding. To ensure our agents weren't just "talking a good game," we built the **Evidence Oracle.** This deterministic component of our reward engine (`reward.py`) performs two critical checks:

1.  **Keyword Saliency**: It matches the agent's summary against a set of "Gold Keywords" specific to the task (e.g., `mutex`, `race condition`, `SQLi`).
2.  **Structural Anchoring**: It verifies that the agent has cited the correct file paths and function names actually present in the provided diff. 

This prevents the model from generating plausible-sounding but generic advice, forcing it to maintain a strict "World Model" of the code change it is evaluating.

## 8. The Curriculum: How We Balanced 100 Scenarios

Training on 100 diverse tasks is a balancing act. To prevent the model from getting overwhelmed by "Hard" tasks early on, our `train_grpo.py` script implements a **Curriculum Sampler.** 

During the initial epochs, the sampler favors **Easy** tasks (60% weight) to establish a baseline of "Good Software Citizenship"—correct formatting, polite tone, and basic logic. As the model's average reward increases, the sampler shifts its weight towards **Medium** (30%) and **Hard** (10%) tasks, ensuring the model builds the necessary complexity in its reasoning layers without losing its foundational accuracy.

---

## 9. Getting Started: Developer Quickstart

The entire protocol is designed for extensibility. You can interact with the environment directly via the API:

### 1. Reset the Environment
```bash
curl -X POST http://127.0.0.1:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task": "hard"}'
```

### 2. Submit an Action
```bash
curl -X POST http://127.0.0.1:7860/step \
     -H "Content-Type: application/json" \
     -H "session_id: YOUR_UUID" \
     -d '{
       "decision": "request_changes",
       "labels": ["bug", "urgent"],
       "priority": "high",
       "review_summary": "Identified TOCTOU race condition in Redis rate limiter logic."
     }'
```

---

## 10. Looking Ahead: The Future of Autonomous Review

`pr-review-env` is the foundation. Our roadmap for the next phase includes:
1.  **Multi-Agent Negotiation**: Having two agents (one "Author Advocate," one "Security Auditor") argue over the PR to reach a final triage decision.
2.  **Live GitHub Integration**: Moving from static fixtures to a live "shadow-mode" on open-source repos.
3.  **Static Analysis "Oracle"**: Coupling the reward system with real tools like Semgrep to provide an objective "ground truth."

By turning code review from a subjective art into a measurable, trainable science, we are building the foundation for the next generation of software engineering intelligence.

---

*Explore the Protocol:* [meta/pr-review-env](https://github.com/...)
*Experience the Triage:* [Hugging Face Space](https://huggingface.co/spaces/...)
