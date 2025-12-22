"""
TF-IDF ranking module.
Scores documents using a simple TF-IDF dot product.
"""
import math
from collections import Counter, defaultdict

from .tokenizer import Tokenizer


class TFIDFRanker:
    """TF-IDF scoring ranker."""

    def __init__(self, index):
        self.index = index
        self.tokenizer = Tokenizer()

    def _calc_idf(self, term):
        """Compute IDF with smoothing."""
        total_docs = self.index.total_docs or 0
        df = self.index.get_doc_freq(term)
        return math.log((total_docs + 1) / (df + 1)) + 1.0

    @staticmethod
    def _calc_tf(freq):
        """Compute log-scaled TF."""
        if freq <= 0:
            return 0.0
        return 1.0 + math.log(freq)

    def score(self, query, top_k=100):
        """
        Score documents for a query using TF-IDF.

        Returns: [(doc_id, score), ...] sorted by score desc.
        """
        terms = self.tokenizer.tokenize(query)
        if not terms:
            return []

        query_tf = Counter(terms)
        query_weights = {term: self._calc_tf(freq) for term, freq in query_tf.items()}

        doc_scores = defaultdict(float)
        for term, q_weight in query_weights.items():
            posting = self.index.get_posting(term)
            if not posting:
                continue

            idf = self._calc_idf(term)
            for doc_id, tf in posting:
                doc_tf = self._calc_tf(tf)
                doc_scores[doc_id] += (doc_tf * idf) * q_weight

        for doc_id in list(doc_scores.keys()):
            doc_len = self.index.doc_len.get(doc_id, 0) or 1
            doc_scores[doc_id] /= doc_len

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
