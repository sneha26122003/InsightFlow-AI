from collections import Counter
import re

def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def rouge_n(hypothesis, reference, n=1):
    hyp = get_ngrams(tokenize(hypothesis), n)
    ref = get_ngrams(tokenize(reference), n)
    overlap = sum((hyp & ref).values())
    precision = overlap / max(sum(hyp.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {"precision": round(precision,4),
            "recall": round(recall,4),
            "f1": round(f1,4)}

def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1]==y[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(hypothesis, reference):
    h = tokenize(hypothesis)
    r = tokenize(reference)
    lcs = lcs_length(h, r)
    p = lcs / max(len(h), 1)
    rec = lcs / max(len(r), 1)
    f1 = (2*p*rec) / max(p+rec, 1e-8)
    return {"precision": round(p,4),
            "recall": round(rec,4),
            "f1": round(f1,4)}

def evaluate_summary(hypothesis, reference):
    r1 = rouge_n(hypothesis, reference, 1)
    r2 = rouge_n(hypothesis, reference, 2)
    rl = rouge_l(hypothesis, reference)
    score = rl["f1"]
    if score >= 0.5:   quality = "Excellent"
    elif score >= 0.4: quality = "Good"
    elif score >= 0.3: quality = "Fair"
    else:              quality = "Needs Improvement"
    return {"rouge_1": r1, "rouge_2": r2, "rouge_l": rl,
            "overall_score": round(score,4), "quality": quality}