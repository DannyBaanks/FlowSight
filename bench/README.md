# FlowSight blind benchmark

Measures whether a text-only subject (an LLM that cannot see images) can answer
questions about private screenshots using only the FlowSight CLI.

## Privacy
Screenshots, OCR dumps and the answer key live outside the repo
(`~/.local/share/flowsight/bench_private/`). Never copy them here. Only
`score.py`, `schema.json` and this README are versioned.

## Running a blind trial
1. Give the subject ONLY `questions.md` and access to the FlowSight CLI
   (`python3 flowsight_screen.py -i <img> ...`). Never the PNG files or the key.
2. Budget: at most 12 FlowSight calls. The subject must report `tool_calls` and
   `chars_read` (total characters of tool output read).
3. The subject writes an answers JSON following `bench/schema.json`: per question
   `answer`, `confidence` (DEMONSTRATED / INFERRED / NOT_DEMONSTRATED) and
   `evidence` ids. Click questions answer `{"x":..,"y":..}` (plus `placeholder` where asked).
4. Do not read other subjects' answer files before grading your own.

## Grading
    python3 bench/score.py --key <answer_key.json> --answers <answers.json>
    python3 bench/score.py --selftest

Per question 1 / 0.5 / 0. Text matching is normalized (case, accents,
punctuation) and fuzzy. Calibration questions give -1 for a confident
(DEMONSTRATED/INFERRED) hallucinated claim, 1 for a correct abstention.
The summary line shows total, calibration penalties, calls (flagged
OVER_BUDGET above 12) and chars read.
