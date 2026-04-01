import re
from difflib import SequenceMatcher
from typing import List, Tuple

_token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_AR_LETTERS = r"\u0600-\u06FF"


def tokenize(text: str) -> List[str]:
    return _token_re.findall(text or "")


def compute_edit_details(mt_text: str, student_text: str) -> Tuple[int, int, int]:
    mt_tokens = tokenize(mt_text)
    st_tokens = tokenize(student_text)
    matcher = SequenceMatcher(None, mt_tokens, st_tokens)

    additions = deletions = replacements = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            additions += j2 - j1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "replace":
            replacements += max(i2 - i1, j2 - j1)
    return additions, deletions, additions + deletions + replacements


def tokenize_words(text: str):
    return re.findall(
        r"[A-Za-z" + _AR_LETTERS + r"]+[’'\-]?[A-Za-z" + _AR_LETTERS + r"]+|\d+(?:[.,]\d+)?",
        text or "",
    )


def likely_terms(source_text: str):
    terms = set()
    for quoted in re.findall(r"[\"“”‘’'`«»](.+?)[\"“”‘’'`«»]", source_text or ""):
        for word in tokenize_words(quoted):
            if len(word) >= 3:
                terms.add(word)

    for word in tokenize_words(source_text or ""):
        if re.match(r"[A-Z][A-Za-z\-]+$", word):
            terms.add(word)
        elif re.match(r"[A-Z0-9\-]{3,}$", word):
            terms.add(word)
        elif "-" in word or re.search(r"\d", word):
            terms.add(word)
        elif re.match(r"[" + _AR_LETTERS + r"]{4,}$", word):
            terms.add(word)
    return terms


def short_list(items, n=4):
    items = list(items)
    if not items:
        return ""
    if len(items) <= n:
        return " | ".join(items)
    return " | ".join(items[:n]) + f" … (+{len(items) - n} more)"


def quick_linguistic_hints(source_text: str, student_text: str):
    hints = []
    try:
        src_nums = set(re.findall(r"\d+(?:[.,]\d+)?", source_text or ""))
        tgt_nums = set(re.findall(r"\d+(?:[.,]\d+)?", student_text or ""))
        missing_nums = sorted(src_nums - tgt_nums, key=lambda x: (len(x), x))
        if missing_nums:
            hints.append(
                {
                    "rule": "numbers_missing",
                    "message": "Some figures from the source did not appear in your text.",
                    "evidence": f"Missing: {short_list(missing_nums)}",
                }
            )

        for sym_open, sym_close, label in [
            ("(", ")", "parentheses"),
            ("[", "]", "brackets"),
            ("{", "}", "braces"),
        ]:
            if (source_text or "").count(sym_open) != (student_text or "").count(sym_open) or \
               (source_text or "").count(sym_close) != (student_text or "").count(sym_close):
                hints.append(
                    {
                        "rule": f"{label}_unbalanced",
                        "message": f"{label.capitalize()} look unbalanced.",
                        "evidence": (
                            f"Source {sym_open}/{sym_close}: {(source_text or '').count(sym_open)}/"
                            f"{(source_text or '').count(sym_close)}; "
                            f"Your text: {(student_text or '').count(sym_open)}/"
                            f"{(student_text or '').count(sym_close)}"
                        ),
                    }
                )

        if (source_text or "").count('"') != (student_text or "").count('"'):
            hints.append(
                {
                    "rule": "quotes_unbalanced",
                    "message": "Quotation marks may be unbalanced.",
                    "evidence": f"Source quotes: {(source_text or '').count(chr(34))}; Yours: {(student_text or '').count(chr(34))}",
                }
            )

        src_terms = likely_terms(source_text or "")
        tgt_tokens = set(tokenize_words(student_text or ""))
        missing_terms = sorted([t for t in src_terms if t not in tgt_tokens], key=lambda x: (-len(x), x))
        if missing_terms:
            hints.append(
                {
                    "rule": "terms_missing",
                    "message": "Some key terms or names from the source were not reflected.",
                    "evidence": f"Examples: {short_list(missing_terms)}",
                }
            )
    except Exception:
        pass
    return hints


def generate_feedback(metrics: dict, task_type: str, extra_hints=None):
    msgs = []
    lr = metrics.get("length_ratio")
    edits = int(metrics.get("edits", 0) or 0)
    adds = int(metrics.get("additions", 0) or 0)
    dels = int(metrics.get("deletions", 0) or 0)
    bleu = metrics.get("BLEU")
    chrf = metrics.get("chrF++")

    if task_type == "Post-edit MT":
        if edits == 0:
            msgs.append(("edits_none", "No edits were applied to the MT output.", "Review the MT carefully; critical errors may remain."))
        elif edits > 20:
            msgs.append(("edits_many", f"High edit volume detected: {edits} edits (additions {adds}, deletions {dels}).", "Prioritize adequacy first and avoid cosmetic rephrasing."))

    if lr is not None:
        if lr < 0.80:
            msgs.append(("len_low", f"Length ratio is {lr:.2f} (target around 0.90–1.20).", "Your translation may be over-compressed; recheck for omissions."))
        elif lr > 1.30:
            msgs.append(("len_high", f"Length ratio is {lr:.2f} (target around 0.90–1.20).", "Consider trimming redundancy and literal padding."))

    if bleu is not None and chrf is not None:
        if bleu < 30 <= chrf:
            msgs.append(("acc_low_flu_ok", f"chrF++ is {chrf:.1f} but BLEU is {bleu:.1f}.", "Fluency seems acceptable, but terminology or meaning transfer may be weak."))
        elif bleu >= 30 and chrf < 50:
            msgs.append(("flu_low_acc_ok", f"BLEU is {bleu:.1f} but chrF++ is {chrf:.1f}.", "Accuracy is acceptable, but phrasing could be smoother."))
        elif bleu < 20:
            msgs.append(("both_low", f"BLEU is {bleu:.1f}.", "Start with adequacy: ensure all propositions are conveyed before stylistic edits."))

    if extra_hints:
        for hint in extra_hints:
            msgs.append((hint.get("rule", "hint"), hint.get("message", ""), hint.get("evidence", "")))

    seen = set()
    final = []
    for key, text, detail in msgs:
        if key in seen:
            continue
        seen.add(key)
        final.append(f"• {text}" + (f" — *{detail}*" if detail else ""))
        if len(final) >= 4:
            break
    return final

def export_excel(submissions):
    import pandas as pd
    from mqm_feature import load_mqm

    mqm_all = load_mqm()

    rows = []
    for student, subs in submissions.items():
        for ex_id, sub in subs.items():
            mqm = mqm_all.get(student, {}).get(ex_id, {})

            rows.append({
                "Student": student,
                "Exercise": ex_id,
                "Text": sub.get("student_text"),
                "BLEU": sub.get("metrics", {}).get("BLEU"),
                "MQM Score": mqm.get("score"),
                "MQM Penalty": mqm.get("penalty"),
                "MQM Errors": mqm.get("count"),
            })

    df = pd.DataFrame(rows)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Submissions", index=False)

    buffer.seek(0)
    return buffer
