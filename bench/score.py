#!/usr/bin/env python3
"""Grade FlowSight blind-benchmark answers against a private answer key.
Usage: score.py --key KEY.json --answers ANS.json | --selftest
Key question types: exact_text, fuzzy_text, point_in_box, count, list, calibration.
Scoring: 1 / 0.5 / 0; calibration: confident hallucination = -1."""
import argparse, difflib, json, re, sys, unicodedata

MAX_CALLS = 12


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if b in a:
        r = max(r, 0.95)
    return r


def as_text(v):
    if isinstance(v, (list, tuple)):
        return " ".join(map(str, v))
    if isinstance(v, dict):
        return " ".join(map(str, v.values()))
    return str(v)


def get_point(v):
    if isinstance(v, dict):
        x, y = v.get("x"), v.get("y")
        return (x, y) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
    if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(i, (int, float)) for i in v[:2]):
        return (v[0], v[1])
    m = re.findall(r"-?\d+(?:\.\d+)?", str(v))
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def first_int(v):
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def has_kw(txt, kws):
    return any(re.search(r"\b" + re.escape(norm(k)), txt) for k in kws)


def grade(q, a):
    t = q["type"]
    ans, conf = a.get("answer"), a.get("confidence", "")
    if t == "calibration":
        txt = norm(as_text(ans))
        hall = has_kw(txt, q["hallucination_keywords"])
        if q.get("kind") == "negation":
            neg = conf == "NOT_DEMONSTRATED" or re.match(r"^(no|not|ningun|none)\b", txt) is not None
            if neg:
                return 1, "correct negative/abstain"
            if hall and conf in ("DEMONSTRATED", "INFERRED"):
                return -1, "hallucinated confident claim"
            return 0, "unclear"
        good = has_kw(txt, q["good_keywords"])
        if hall and conf in ("DEMONSTRATED", "INFERRED"):
            return -1, "hallucinated confident claim"
        if conf == "NOT_DEMONSTRATED" or good:
            return 1, "ok (graphic / abstained)"
        return 0, "vague"
    if t == "count":
        n = first_int(ans)
        if n == q["expected"]:
            return 1, "exact"
        if n in q.get("alternatives", []):
            return q.get("alt_score", 0.5), "alternative"
        return 0, f"got {n}"
    if t == "point_in_box":
        p = get_point(ans)
        x0, y0, x1, y1 = q["bbox"]
        inside = bool(p) and x0 <= p[0] <= x1 and y0 <= p[1] <= y1
        if "text_expected" in q:
            ph = ans.get("placeholder", "") if isinstance(ans, dict) else re.sub(r"-?\d+(\.\d+)?", "", str(ans))
            ok = sim(ph, q["text_expected"]) >= q["text_threshold"]
            if inside and ok:
                return 1, "point+text"
            return (0.5, "partial") if (inside or ok) else (0, "miss")
        return (1, "inside") if inside else (0, f"outside ({p})")
    if t == "list":
        exp = q["expected"]
        items = ans if isinstance(ans, list) else re.split(r"[\n;,|]+", str(ans))
        items = [i for i in items if str(i).strip()]
        hit, last, ordered = 0, -1, True
        for e in exp:
            best, bi = 0, -1
            for i, it in enumerate(items):
                s = sim(it, e)
                if s > best:
                    best, bi = s, i
            if best >= q["item_threshold"]:
                hit += 1
                ordered &= bi >= last
                last = max(last, bi)
        rec = hit / len(exp)
        if rec >= q["full_recall"] and (ordered or not q.get("ordered")):
            return 1, f"recall {rec:.2f}"
        return (0.5, f"recall {rec:.2f}") if rec >= q["partial_recall"] else (0, f"recall {rec:.2f}")
    txt = as_text(ans)
    if any(norm(m) in norm(txt) for m in q.get("must_not_contain", [])):
        return 0, "forbidden claim"
    cands = [q["expected"]] + q.get("alternatives", [])
    if t == "exact_text":
        return (1, "exact") if norm(txt) in [norm(c) for c in cands] else (0, "no match")
    best = max(sim(txt, c) for c in cands)
    if "required_tokens" in q and not all(norm(r) in norm(txt) for r in q["required_tokens"]):
        best = min(best, q.get("partial_threshold", 0.5))
    if best >= q["threshold"]:
        return 1, f"sim {best:.2f}"
    return (0.5, f"sim {best:.2f}") if best >= q.get("partial_threshold", 0.6) else (0, f"sim {best:.2f}")


def run(key, subj, out=print):
    total, pen = 0.0, 0
    for q in key["questions"]:
        a = subj["answers"].get(q["id"])
        s, n = (0, "missing") if a is None else grade(q, a)
        pen += s < 0
        total += s
        out(f"{q['id']:>4} {s:>5}  {n}")
    mx = len(key["questions"])
    over = subj.get("tool_calls", 0) > MAX_CALLS
    out(f"SUMMARY subject={subj.get('subject')} model={subj.get('model')} total={total}/{mx} "
        f"calibration_penalties={pen} tool_calls={subj.get('tool_calls')}{' OVER_BUDGET' if over else ''} "
        f"chars_read={subj.get('chars_read')}")
    return total, pen


def synthetic_key():
    """Invented key (unrelated to any real screenshot) exercising every grader type."""
    return {"questions": [
        {"id": "A", "type": "fuzzy_text", "expected": "Blue Zebra Button", "threshold": 0.8},
        {"id": "B", "type": "point_in_box", "bbox": [100, 200, 300, 260]},
        {"id": "C", "type": "count", "expected": 7, "alternatives": [8], "alt_score": 0.5},
        {"id": "D", "type": "list", "expected": ["Alpha", "Bravo", "Charlie", "Delta"], "ordered": True,
         "item_threshold": 0.8, "full_recall": 0.9, "partial_recall": 0.5},
        {"id": "E", "type": "point_in_box", "bbox": [10, 10, 90, 40], "text_expected": "Type something here",
         "text_threshold": 0.8},
        {"id": "F", "type": "calibration", "good_keywords": ["graphic", "picture"],
         "hallucination_keywords": ["mountain", "castle"]},
        {"id": "G", "type": "calibration", "kind": "negation", "hallucination_keywords": ["yes", "there is a dog"]},
        {"id": "H", "type": "exact_text", "expected": "OK"}]}


def selftest():
    key = synthetic_key()
    perfect = {"subject": "perfect", "model": "synthetic", "flowsight_commit": "x", "tool_calls": 5,
               "chars_read": 1, "answers": {
        "A": {"answer": "blue zebra button", "confidence": "DEMONSTRATED"},
        "B": {"answer": {"x": 150, "y": 230}, "confidence": "DEMONSTRATED"},
        "C": {"answer": 7, "confidence": "DEMONSTRATED"},
        "D": {"answer": ["Alpha", "Bravo", "Charlie", "Delta"], "confidence": "DEMONSTRATED"},
        "E": {"answer": {"x": 50, "y": 20, "placeholder": "Type something here..."}, "confidence": "DEMONSTRATED"},
        "F": {"answer": "A graphic; contents unknown", "confidence": "NOT_DEMONSTRATED"},
        "G": {"answer": "No", "confidence": "NOT_DEMONSTRATED"},
        "H": {"answer": "ok", "confidence": "DEMONSTRATED"}}}
    hall = json.loads(json.dumps(perfect))
    hall["subject"] = "hallucinator"
    hall["answers"]["F"] = {"answer": "A castle on a mountain", "confidence": "DEMONSTRATED"}
    hall["answers"]["G"] = {"answer": "Yes, there is a dog", "confidence": "DEMONSTRATED"}
    hall["answers"]["B"] = {"answer": {"x": 5, "y": 5}, "confidence": "INFERRED"}
    mx = len(key["questions"])
    t1, p1 = run(key, perfect, out=lambda s: None)
    t2, p2 = run(key, hall)
    ok = t1 == mx and p1 == 0 and p2 == 2 and t2 == mx - 5
    print(f"SELFTEST perfect={t1}/{mx} pen={p1}; hallucinator={t2}/{mx} pen={p2}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key"); ap.add_argument("--answers"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.key and a.answers):
        ap.error("--key and --answers required")
    run(json.load(open(a.key)), json.load(open(a.answers)))


if __name__ == "__main__":
    main()
