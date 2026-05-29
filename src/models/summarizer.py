from transformers import pipeline
import torch
import re
from heapq import nlargest
from collections import defaultdict
from typing import Literal
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
LENGTH_PRESETS = {
    "short":  {"min_length": 60,  "max_length": 150},
    "medium": {"min_length": 150,  "max_length": 350},
    "long":   {"min_length": 300, "max_length": 600},
}

STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and",
    "or","but","with","this","that","are","was","were","be","been",
    "have","has","do","does","did","will","ka","ki","ke","hai","hain",
    "mein","se","ko","ne","aur","ya","lekin","par","bhi","hi","jo"
}

class TextSummarizer:
    def __init__(self, model_name: Literal["bart","t5"] = "bart"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        names = {
            "bart": "facebook/bart-large-cnn",
            "t5":   "t5-base",
        }
        self.model_id = names[model_name]
        self.prefix = "summarize: " if model_name == "t5" else ""
        print(f"Loading {self.model_id} on {self.device}...")
        self.pipe = pipeline(
            "summarization",
            model=self.model_id,
            device=0 if self.device=="cuda" else -1,
        )
        print("Model ready!")

    def summarize(self, text: str, length: str = "medium") -> dict:
        if len(text.split()) < 30:
            raise ValueError("Text bahut chhota hai (min 30 words)")
        params = LENGTH_PRESETS[length]
        out = self.pipe(
            self.prefix + text,
            min_length=params["min_length"],
            max_length=params["max_length"],
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        summary = out[0]["summary_text"]
        orig = len(text.split())
        summ = len(summary.split())
        return {
            "summary": summary,
            "original_length": orig,
            "summary_length": summ,
            "compression_ratio": round((1 - summ/orig)*100, 1),
            "model_used": self.model_id,
        }


class ExtractiveSummarizer:
    def summarize(self, text: str, num_sentences: int = 3) -> dict:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= num_sentences:
            return {"summary": text, "compression_ratio": 0.0}

        freq = defaultdict(int)
        for s in sentences:
            for w in re.sub(r'[^a-zA-Z ]','',s).lower().split():
                if w not in STOPWORDS and len(w) > 2:
                    freq[w] += 1
        max_f = max(freq.values(), default=1)
        for w in freq:
            freq[w] /= max_f

        scores = {}
        for i, s in enumerate(sentences):
            words = re.sub(r'[^a-zA-Z ]','',s).lower().split()
            wc = max(len(words), 1)
            scores[i] = sum(freq.get(w,0) for w in words) / wc

        top = sorted(nlargest(num_sentences, scores, key=scores.get))
        summary = " ".join(sentences[i] for i in top)
        ratio = round((1 - num_sentences/len(sentences))*100, 1)
        return {
            "summary": summary,
            "compression_ratio": ratio,
            "sentences_selected": num_sentences,
            "total_sentences": len(sentences),
        }