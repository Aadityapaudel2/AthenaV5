# POLISH LOG — x3 (canonical)
Scope: verbosity reduction + metadata/text normalization (no merges/compilation).

## Inputs / Outputs
- Input: `bibletrainingdata_canonical_polished_x1.jsonl` (36502 lines)
- Output: `bibletrainingdata_canonical_polished_x3.jsonl` (36502 lines)

## Deterministic Normalizations Applied
1. **Typography normalization**: curly quotes/apostrophes → ASCII, ellipsis → `...`, NBSP → space.
2. **Prefix compression** (token savings; meaning preserved):
   - `Athena records the passage under Neohm's horizon:` → `Athena:` (hits: 3016)
   - `Neohm speaks into the record:` → `Neohm:` (hits: 11166)
   - `A scholion clarifies the scene:` → `Scholion:` (hits: 624)
   - `Read this as training:` → `Training:` (hits: 1072)
   - `Prompt takeaway:` → `Takeaway:` (hits: 17840)
   - `Neohm sets a drill for Arjuna:` → `Drill — Neohm→Arjuna:` (hits: 524)
3. **Label normalization**: `Prompt takeaway:` → `Takeaway:` (hits counted above).
4. **Publisher boilerplate removal**: stripped accidental Project Gutenberg appendix/license text where detected.

## Change Counters
- Lines with any content change: 18095
- Lines with typography normalization: 1637
- Lines with whitespace collapse: 2
- Lines with Gutenberg boilerplate stripped: 1
- Exact duplicates removed (post-transform): 0 (expected 0 or low; none detected)

## Safety invariants preserved
- `Neohm-Detected` tag preserved (count unchanged).
- No banned forms reintroduced (Landneohm / Neohm Neohm / Neohm Jesus* = 0).
- `Lord` token preserved; no deity-title corruption performed in x3.

## Record length sanity check
- Pre-x3 max teacher record length was dominated by accidental Gutenberg boilerplate; post-x3 max teacher record length is within normal intro/prologue bounds.