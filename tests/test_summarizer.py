import pytest
from src.models.summarizer import ExtractiveSummarizer
from src.utils.evaluation import rouge_n, rouge_l, evaluate_summary

LONG_TEXT = """
Artificial intelligence has transformed how we interact with technology.
Modern AI systems can understand natural language and generate content.
These advances have created new opportunities in healthcare and education.
Researchers are working to ensure AI development proceeds responsibly.
The next decade promises even more dramatic advances as compute grows.
"""

class TestExtractiveSummarizer:
    def setup_method(self):
        self.s = ExtractiveSummarizer()

    def test_basic(self):
        r = self.s.summarize(LONG_TEXT, 2)
        assert "summary" in r
        assert len(r["summary"]) > 0

    def test_compression(self):
        r = self.s.summarize(LONG_TEXT, 2)
        assert 0 <= r["compression_ratio"] <= 100

class TestROUGE:
    def test_perfect(self):
        t = "the cat sat on the mat"
        assert rouge_n(t, t, 1)["f1"] == 1.0

    def test_no_match(self):
        assert rouge_n("cat sat", "python rocks", 1)["f1"] == 0.0

    def test_rougel_partial(self):
        s = rouge_l("the cat sat on a mat",
                    "the cat sat on the mat outside")
        assert 0 < s["f1"] < 1.0

    def test_all_metrics(self):
        r = evaluate_summary("AI is powerful",
                             "AI is very powerful technology")
        for k in ["rouge_1","rouge_2","rouge_l",
                  "overall_score","quality"]:
            assert k in r