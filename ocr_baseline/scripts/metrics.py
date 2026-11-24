"""NFC only; micro Levenshtein CER/WER, never remove whitespace/punctuation."""
import unicodedata


def distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def measure(gt, prediction):
    gt = unicodedata.normalize('NFC', gt)
    prediction = unicodedata.normalize('NFC', prediction)
    numeric = any(c.isdigit() for c in gt) and not any(
        ('LATIN' in unicodedata.name(c, '') or 'HANGUL' in unicodedata.name(c, ''))
        and unicodedata.category(c).startswith('L') for c in gt)
    return dict(ground_truth_norm=gt, prediction_norm=prediction,
                char_edits=distance(gt, prediction), gt_chars=len(gt),
                word_edits=distance(gt.split(), prediction.split()), gt_words=len(gt.split()),
                exact_match=int(gt == prediction), numeric_symbol_only=int(numeric))


def aggregate(rows):
    rows = list(rows)
    totals = {key: sum(int(r[key]) for r in rows) for key in ('char_edits', 'gt_chars', 'word_edits', 'gt_words')}
    numeric = [r for r in rows if int(r['numeric_symbol_only'])]
    return dict(n=len(rows), cer_pct=100 * totals['char_edits'] / totals['gt_chars'] if totals['gt_chars'] else None,
                wer_pct=100 * totals['word_edits'] / totals['gt_words'] if totals['gt_words'] else None,
                exact_match_pct=100 * sum(int(r['exact_match']) for r in rows) / len(rows) if rows else None,
                numeric_symbol_only_n=len(numeric),
                numeric_symbol_only_exact_pct=100 * sum(int(r['exact_match']) for r in numeric) / len(numeric) if numeric else None,
                **totals)
