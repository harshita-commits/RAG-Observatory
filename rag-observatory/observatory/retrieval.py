"""Inspectable sparse retrieval baselines; fitted exclusively on the corpus."""
import math
import heapq
import re
from collections import Counter

STOP = set("the a an is are of for to and in on what how do does can i it my with when which who as at be".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class Retriever:
    def __init__(self, documents):
        self.docs = documents
        self.counts = [Counter(tokens(d["text"])) for d in documents]
        self.df = Counter(w for c in self.counts for w in c)
        self.n = len(documents)
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_len = sum(self.lengths) / self.n or 1
        self.idf = {w: math.log((1+self.n)/(1+df))+1 for w, df in self.df.items()}
        self.weights = [{w: count*self.idf[w] for w, count in c.items()} for c in self.counts]
        self.norms = [math.sqrt(sum(v*v for v in weights.values())) for weights in self.weights]
        self.postings = {}
        for index, counts in enumerate(self.counts):
            for word, count in counts.items():
                self.postings.setdefault(word, []).append((index, count))

    def scores(self, words, strategy):
        values = Counter()
        if strategy == "tfidf":
            query = {word: count*self.idf.get(word, math.log(1+self.n)+1) for word, count in words.items()}
            norm = math.sqrt(sum(value*value for value in query.values()))
            for word, weight in query.items():
                for index, count in self.postings.get(word, []):
                    values[index] += weight*count*self.idf[word]
            return {index: dot/(norm*self.norms[index]) for index, dot in values.items() if norm and self.norms[index]}
        for word in words:
            inverse = math.log(1+(self.n-self.df[word]+.5)/(self.df[word]+.5))
            for index, frequency in self.postings.get(word, []):
                values[index] += inverse*frequency*2.5/(frequency+1.5*(.25+.75*self.lengths[index]/self.avg_len))
        return dict(values)

    def search(self, query, strategy="bm25", top_k=3):
        if strategy not in {"tfidf", "bm25", "rrf"}:
            raise ValueError("strategy must be tfidf, bm25 or rrf")
        if not 1 <= top_k <= len(self.docs):
            raise ValueError("top_k is outside the corpus bounds")
        words = Counter(tokens(query))
        if strategy == "rrf":
            fused = Counter()
            for component in ("tfidf", "bm25"):
                scores = self.scores(words, component)
                ranking = sorted(scores, key=lambda index: (-scores[index], self.docs[index]["id"]))
                for rank, index in enumerate(ranking, 1): fused[index] += 1/(60+rank)
            scores = dict(fused)
        else:
            scores = self.scores(words, strategy)
        ranking = heapq.nsmallest(top_k, scores, key=lambda index: (-scores[index], self.docs[index]["id"]))
        return [{**self.docs[index], "score": round(scores[index], 8)} for index in ranking if scores[index] > 0]


def extract_answer(question, documents):
    """Offline extractive baseline. Never uses the reference answer."""
    words = set(tokens(question))
    sentences = []
    for doc in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", doc["text"]):
            overlap = len(words & set(tokens(sentence))) / max(len(words), 1)
            sentences.append((overlap, sentence, doc["id"]))
    best = sorted(sentences, key=lambda row: -row[0])[:2]
    if not best or best[0][0] < .24:
        return "I do not know based on the available evidence.", []
    selected = [row for row in best if row[0] >= .24]
    return " ".join(f"{s} [{i}]" for _, s, i in selected), list(dict.fromkeys(i for _, _, i in selected))
