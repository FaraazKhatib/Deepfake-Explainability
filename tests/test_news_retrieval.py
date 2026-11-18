import unittest
from unittest.mock import patch

from backend.app.services.text.claims.query_builder import build_query
from backend.app.services.text.retrieval import aggregator as agg
from backend.app.services.text.retrieval.base import Evidence, RetrievalReport
from backend.app.services.text.scoring.explain import short_summary
from backend.app.services.text.scoring.verdict import ClaimVerdict, VERDICT_UNVERIFIED


class NewsRetrievalTests(unittest.TestCase):
    claim = "Japan launches lunar research satellite"
    def item(self, title, domain):
        return Evidence(title, f"https://{domain}/story", domain, agg.news_rss.SOURCE_NAME)

    def retrieve(self, responses):
        with patch.object(agg.config, "ENABLE_GDELT", False), patch.object(agg.config, "ENABLE_WIKIPEDIA", False), patch.object(
            agg.news_rss, "search", side_effect=responses
        ) as search:
            report = agg.gather_evidence(build_query(self.claim), max_items=2, claim_text=self.claim)
        return report, search.call_count

    def test_late_matching_english_headlines_reach_model(self):
        noise = [self.item(f"The council built a new bridge number {i}", f"{i}.example") for i in range(15)]
        hits = [self.item(self.claim, "a.example"), self.item("Japan launches lunar satellite carrying research instruments", "b.example")]
        report, calls = self.retrieve([noise + hits])
        self.assertEqual({e.domain for e in report.evidence}, {"a.example", "b.example"})
        self.assertEqual(calls, 1)

    def test_retry_empty_query_once(self):
        hit = self.item(self.claim, "a.example")
        report, calls = self.retrieve([[], [hit]])
        self.assertEqual(calls, 2)
        self.assertEqual(report.evidence, [hit])

    def test_outage_and_neutral_evidence_have_different_summaries(self):
        report, calls = self.retrieve([RuntimeError("offline")])
        self.assertEqual(calls, 1)
        verdict = ClaimVerdict(self.claim, VERDICT_UNVERIFIED, "None", 0, 0)
        self.assertIn("unavailable", short_summary({}, [verdict], report))
        verdict.evidence_examined = 3
        self.assertIn("Examined 3", short_summary({}, [verdict], RetrievalReport()))


if __name__ == "__main__":
    unittest.main()
