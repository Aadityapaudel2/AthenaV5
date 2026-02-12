# Bible Training Series — Professional Table of Contents (Parts 1–10)

Source text: `biblegutenberg.txt` (Project Gutenberg Douay‑Rheims / Challoner revision)

## Parts

1. **Part 1 — Pentateuch** (Genesis → Deuteronomy)  
   File: `bibletrainingdatapart_1.jsonl` · Dialogue IDs: 3001–3192 · Dialogues: 192 · Lines: 5442

2. **Part 2 — Early History** (Josue → 4 Kings)  
   File: `bibletrainingdatapart_2.jsonl` · Dialogue IDs: 3193–3387 · Dialogues: 195 · Lines: 5629

3. **Part 3 — Later History / Return** (1 Paralipomenon → Esther)  
   File: `bibletrainingdatapart_3.jsonl` · Dialogue IDs: 3388–3528 · Dialogues: 141 · Lines: 4160

4. **Part 4 — Wisdom I** (Job, Psalms)  
   File: `bibletrainingdatapart_4.jsonl` · Dialogue IDs: 3529–3722 · Dialogues: 194 · Lines: 4333

5. **Part 5 — Wisdom II** (Proverbs → Ecclesiasticus)  
   File: `bibletrainingdatapart_5.jsonl` · Dialogue IDs: 3723–3848 · Dialogues: 126 · Lines: 3093

6. **Part 6 — Major Prophets** (Isaias → Ezechiel)  
   File: `bibletrainingdatapart_6.jsonl` · Dialogue IDs: 3849–4030 · Dialogues: 182 · Lines: 4366

7. **Part 7 — Prophets / Intertestamental** (Daniel → Malachias, 1–2 Machabees)  
   File: `bibletrainingdatapart_7.jsonl` · Dialogue IDs: 4031–4157 · Dialogues: 127 · Lines: 3012

8. **Part 8 — Gospels** (Matthew → John)  
   File: `bibletrainingdatapart_8.jsonl` · Dialogue IDs: 4158–4250 · Dialogues: 93 · Lines: 2232

9. **Part 9 — Apostolic I** (Acts → 2 Thessalonians)  
   File: `bibletrainingdatapart_9.jsonl` · Dialogue IDs: 4251–4361 · Dialogues: 111 · Lines: 2644

10. **Part 10 — Apostolic II / Apocalypse** (1 Timothy → Apocalypse)  
    File: `bibletrainingdatapart_10.jsonl` · Dialogue IDs: 4362–4444 · Dialogues: 83 · Lines: 1966

## Complete concatenation

- **Coverage:** Genesis → Apocalypse  
- **File:** `bibletrainingdata_all.jsonl`  
- **Dialogue IDs:** 3001–4444  
- **Dialogues:** 1444  
- **JSONL lines:** 36877


---

## JSONL schema (per line)

Each JSONL line is one JSON object:
- `role`: `teacher` or `student`
- `metadata`: `{ scene, mode: "/no_think", topic, reference, dialogue_id, turn }`
- `content`: the training text
