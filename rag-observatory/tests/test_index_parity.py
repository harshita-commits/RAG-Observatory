import math
import random
import unittest
from collections import Counter
from observatory.retrieval import Retriever, tokens


class IndexParityTests(unittest.TestCase):
    def test_postings_scores_match_full_scan_formulas(self):
        rng = random.Random(42)
        docs = [{"id": str(i), "text": " ".join(f"term{rng.randrange(20)}" for _ in range(30))} for i in range(50)]
        retriever = Retriever(docs)
        query = Counter(tokens("term1 term1 term7 unseen"))
        for strategy in ("bm25", "tfidf"):
            expected = {}
            for i, counts in enumerate(retriever.counts):
                if strategy == "bm25":
                    score = sum(math.log(1+(retriever.n-retriever.df[w]+.5)/(retriever.df[w]+.5))*counts[w]*2.5/(counts[w]+1.5*(.25+.75*sum(counts.values())/retriever.avg_len)) for w in query)
                else:
                    inv = lambda w: math.log((1+retriever.n)/(1+retriever.df[w]))+1
                    q = {w:c*inv(w) for w,c in query.items()}
                    d = {w:c*inv(w) for w,c in counts.items()}
                    dot = sum(v*d.get(w,0) for w,v in q.items())
                    score = dot/math.sqrt(sum(v*v for v in q.values())*sum(v*v for v in d.values()))
                if score > 0: expected[i] = score
            actual = retriever.scores(query, strategy)
            self.assertEqual(set(expected), set(actual))
            for i in expected: self.assertAlmostEqual(expected[i], actual[i], places=12)

    def test_all_stopword_corpus_and_zero_overlap(self):
        retriever = Retriever([{"id":"a", "text":"the and is"}])
        self.assertEqual(retriever.search("unknown", "bm25", 1), [])


if __name__ == "__main__": unittest.main()
