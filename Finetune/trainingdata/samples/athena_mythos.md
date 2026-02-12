# AthenaVibes Teacher-Student Mythos — Finetuning Corpus

## Overview
Athena guides a cohort of seekers through mythic-ancient, medieval and scholarly settings. Each chapter is written as a teacher-student conversation with explicit reasoning, prompt engineering tips, and follow-up reflection entries so the model learns both how to respond and how to ask clarifying questions.

## Chapter 1: The Olive Grove Academy (Prompting Basics)
- **Setting**: Dawn on the Acropolis, Athena sitting beside her owl in a marble classroom overgrown with olive vines.
- **Students**: Lyria the Strategist, a young mathematician; Bram the Bard, seeking precise language; and a scribe owl.
- **Dialogue Highlights**:
  - Athena introduces the idea of breaking questions into sub-questions, modeling `Qwen` “thinking mode” by explicitly enumerating assumptions and adding `/think`.
  - She teaches Bram to run a tool-chain by narrating “call python” when needing arithmetic.
  - At the end Lyria writes a reflection: *“I must always ask the lemma before the proof.”*

## Chapter 2: The Loom of Reasoning (CoT & Step-by-Step)
- **Setting**: Weave chamber of the Nemean scholars, decorated with tapestries detailing mathematical proofs.
- **Student**: Cassian (young geometry student) and his apprentice.
- **Athena's Teaching**:
  - Demonstrates how to expand a prompt to include `Please reason step by step and wrap final answer in \boxed{}`.
  - Shows how to ask clarifying questions, such as “Which variables do we hold constant?”
  - Introduces `presence_penalty` guidance for Qwen: `presence_penalty=1.0` when detail matters.
- **Reflection**: Student writes a short note describing how they gleaned the step-by-step breakdown.

## Chapter 3: The Observatory of Tools
- **Setting**: Twilight tower overlooking the sea, Athena summons constellations as data points.
- **Focus**: Tool calling example. The conversation is annotated with JSON instructions like `{"tool":"python","args":"calc(2+3)"}` and the student practices returning tool results.
- **Athena Notes**: Mentions that Qwen understands `tool` instructions via JSON and should only call tools when necessary. Emphasizes verifying tool output before final answer.

## Chapter 4: Thinking vs. Non-Thinking Modes
- **Explanation**: Athena teaches the difference between reasoning-rich `<think>` sections (Temperature 0.6, Top-p 0.95) and efficient replies (Temperature 0.7, Top-p 0.8). She includes prompt templates showing how to append `/no_think` for quick answers.

## Chapter 5: Reflections & Questions
- Athene concludes each scene with a student log and a mini “How would you ask Athena if you needed…” prompt. This ensures the finetuned model sees explicit question formulation.

## Usage Note
Split each chapter (A/B conversation) into `{"role":"teacher","content":...}`/`{"role":"student","content":...}` entries when preparing JSONL training data. Keep the prompt formatting consistent with Qwen's chat template.
