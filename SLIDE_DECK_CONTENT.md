# Slide Deck Content: The Agentic Triage Protocol

Follow this structure for a professional, 6-7 slide presentation.

---

## Slide 1: Title & Vision
- **Title**: The Agentic Triage Protocol
- **Subtitle**: Training Professional Code Reviewers with GRPO
- **Visual Idea**: A futuristic shield icon overlaying a code diff.
- **Key Message**: Moving from AI summarization to AI code governance.

## Slide 2: The Problem (The Saliency Gap)
- **Point 1**: LLMs write code but don't know how to guard it.
- **Point 2**: Shallow summaries miss deep technical risks (Concurrency, Security).
- **Point 3**: Models "hedge"—approving bad code to maintain "helpfulness."
- **Visual Idea**: A split screen: "What LLMs see (Text)" vs "What Humans need (Impact)."

## Slide 3: The Solution (pr-review-env)
- **Feature 1**: 100 Scenarios (TOCTOU, SQLi, Saga patterns, N+1 queries).
- **Feature 2**: Staged cognitive workflow (Identify -> Assess -> Triage).
- **Feature 3**: High-fidelity, deterministic OpenEnv backend.
- **Visual Idea**: Icons representing different error categories (Shield, Clock, Database).

## Slide 4: The Reward Engine (Verifiable Logic)
- **Factor 1**: Decision & Label Precision (30% + 25%).
- **Factor 2**: Evidence Oracle (25%) - Deterministic keyword/file verification.
- **Factor 3**: Consistency Penalty - Ending model contradictions.
- **Factor 4**: Latency-Aware Scoring - Accuracy meets CI/CD speed.

## Slide 5: The RL Strategy (GRPO + Unsloth)
- **Method**: Group Relative Policy Optimization.
- **Logic**: Self-play reasoning through group discourse—no separate critic model needed.
- **Efficiency**: 4-bit LoRA via Unsloth.
- **Outcome**: 2-3x faster training and sub-8s inference.

## Slide 6: The Results (The Reasoning Leap)
- **Hard Task Quality**: +77% improvement.
- **Logic Alignment**: +67% in consistency.
- **Inference Speed**: 45% faster reviews.
- **Visual Idea**: A bar chart showing Baseline vs. GRPO for "Hard Tasks."

## Slide 7: Conclusion & Future
- **Pillar 1**: Fully reproducible (Docker + Colab).
- **Pillar 2**: Real-world ready (Latency-optimized).
- **Vision**: Autonomous code observability in every CI/CD pipeline.
- **Links**: GitHub Repo & Hugging Face Space.
