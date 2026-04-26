# The Agentic Triage Protocol: 2-Minute Presentation Script

**Goal**: Deliver a high-impact, professional summary of the project for hackathon judges.

---

### [0:00 - 0:30] The Hook: The Governance Gap
"Current AI coding assistants are great at writing code, but they are dangerously naive at **guarding** it. Most LLMs can summarize a diff, but they can’t tell you if a subtle shift in a Redis lock will bring down your global service. There is a massive 'Governance Gap' between code generation and professional code review."

### [0:30 - 1:00] The Solution: pr-review-env
"We built **`pr-review-env`**—not just a dataset, but a high-fidelity, interactive protocol for training professional reviewers. We've cataloged **100 diversified scenarios** cross-referenced from real production incidents—covering everything from JWT security bypasses to TOCTOU race conditions.
Our environment uses a **Staged Interaction Protocol**, forcing agents to transition from identifying risks to assessing impact and final remediation."

### [1:00 - 1:30] The Tech: GRPO & The Reward Engine
"To solve this, we used **Group Relative Policy Optimization (GRPO)**. We trained a Qwen-2.5-3B model using a deterministic, multi-axis reward engine. We don't just score if the model is 'polite'; we score on **Decision Accuracy, Label F1, and Logical Consistency.** 
Our engine include an **'Anti-Hedging' penalty**—if a model spots a security bug but approves it anyway, it is heavily penalized. This builds a model engineering teams can actually trust."

### [1:30 - 2:00] The Results: Speed & Stability
"The results? A **77% improvement** in triage quality on hard, high-stakes tasks. But more importantly, we’ve broken the speed-stability tradeoff. Our agents now deliver senior-grade reviews in **under 8 seconds**—faster than any human could parse the complexity of these diffs.
We've made this fully reproducible with a Dockerized environment and a Colab-first training pipeline. We aren't just summarizing code; we're automating professional code observability. Thank you."

---

### Key Differentiators to Emphasize (If asked):
1. **Deterministic Grading**: No high-latency LLM judges; all rewards are computed via a fast, verifiable Evidence Oracle.
2. **Latency-Aware RL**: Our RL loop specifically optimizes for the Pareto frontier of accuracy vs. speed.
3. **Consistency Penalty**: We effectively solved "model hedging," where models choose the easiest path instead of the safest one.
