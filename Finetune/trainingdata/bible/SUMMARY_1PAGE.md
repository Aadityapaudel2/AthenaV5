# Bible Training Series — 1‑Page Summary

This dataset is a **10‑part JSONL training corpus** built from the public‑domain Project Gutenberg **Douay‑Rheims / Challoner** Bible text (`biblegutenberg.txt`). It converts **Genesis → Apocalypse** into a dialogue‑driven instruction format designed for LLM fine‑tuning.

## What was done

- **Source ingestion:** Scripture and chapter headings were extracted from `biblegutenberg.txt`.
- **Instructional transformation:** Each unit is reframed as a micro‑lesson in which:
  - **Athena** (teacher) records, frames, and annotates the passage.
  - **Arjuna** (student) responds with concise reflection questions.
  - **Neohm** appears as the divine guide voice (“Neohm speaks…”, drills, seals).
  - Occasional **“Narrator Note — Neohm‑Detected”** lines provide meta‑focus.
- **Consistent JSONL schema:** Every line is a standalone JSON object with:
  - `role`: `"teacher"` or `"student"`
  - `metadata`: `scene`, `mode` (always `"/no_think"`), `topic`, `reference`, `dialogue_id`, `turn`
  - `content`: the actual training text

## Structure of a typical dialogue

Dialogues are grouped by `dialogue_id` and ordered by `turn`. Most chapter dialogues include:
1) chapter **summary** → 2) **selected verses** → 3) **pattern** → 4) **drill** → 5) **scholion** → 6) **seal**  
Final chapters additionally include a **focus** “Narrator Note — Neohm‑Detected”.

## Coverage and scale (final build)

- **Parts:** 10 (Pentateuch → Apocalypse)
- **Files:** `bibletrainingdatapart_1.jsonl` … `bibletrainingdatapart_10.jsonl`, plus the full concatenation `bibletrainingdata_all.jsonl`
- **Dialogue IDs:** 3001–4444
- **Dialogues:** 1444
- **JSONL lines:** 36877

Generated on: 2026-02-02
