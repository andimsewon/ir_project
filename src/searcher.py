"""
Search engine module.
Supports BM25, TF-IDF, hybrid, and optional reranking/query expansion.
"""
from collections import defaultdict

from .tokenizer import Tokenizer


class SearchEngine:
    """
    Unified search pipeline.

    Args:
        index: InvertedIndex instance.
        bm25_ranker: BM25Ranker instance.
        reranker: optional reranker.
        tfidf_ranker: optional TFIDFRanker instance.
        query_expander: optional QueryExpander instance.
        dense_retriever: optional DenseRetriever instance.
        splade_retriever: optional SpladeRetriever instance.
    """

    def __init__(
        self,
        index,
        bm25_ranker,
        reranker=None,
        tfidf_ranker=None,
        query_expander=None,
        dense_retriever=None,
        splade_retriever=None,
    ):
        self.index = index
        self.bm25_ranker = bm25_ranker
        self.reranker = reranker
        self.tfidf_ranker = tfidf_ranker
        self.query_expander = query_expander
        self.dense_retriever = dense_retriever
        self.splade_retriever = splade_retriever
        self.tokenizer = Tokenizer()

    def search(
        self,
        query,
        top_k=10,
        method="bm25",
        use_reranker=False,
        use_query_expansion=False,
        hybrid_weight=0.6,
        num_candidates=100,
    ):
        """
        Run a search query.

        Returns:
            {
                'query': str,
                'method': str,
                'results': [{'rank', 'doc_id', 'score', 'snippet'}, ...],
                'expanded_query': str (optional)
            }
        """
        expanded_query = query
        if use_query_expansion and self.query_expander:
            expanded_query = self.query_expander.expand(query, method="hybrid")

        candidates, method_name = self._search_by_method(
            expanded_query,
            method=method,
            num_candidates=num_candidates,
            hybrid_weight=hybrid_weight,
        )

        if not candidates:
            result = {'query': query, 'method': method_name, 'results': []}
            if expanded_query != query:
                result['expanded_query'] = expanded_query
            return result

        if use_reranker and self.reranker:
            rerank_query = query
            candidates = self.reranker.rerank(
                rerank_query, candidates, self.index.doc_store, top_k=top_k
            )
            method_name = f"{method_name} + Reranker"
        else:
            candidates = candidates[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(candidates, 1):
            snippet = self._extract_snippet(doc_id, query)
            results.append(
                {
                    'rank': rank,
                    'doc_id': doc_id,
                    'score': score,
                    'snippet': snippet,
                }
            )

        result = {'query': query, 'method': method_name, 'results': results}
        if expanded_query != query:
            result['expanded_query'] = expanded_query
        return result

    def _search_by_method(self, query, method, num_candidates, hybrid_weight):
        method = (method or "bm25").lower()

        if method == "tfidf":
            if self.tfidf_ranker:
                return self.tfidf_ranker.score(query, top_k=num_candidates), "TF-IDF"
            return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

        if method == "hybrid":
            if not self.tfidf_ranker:
                return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

            bm25 = self.bm25_ranker.score(query, top_k=num_candidates)
            tfidf = self.tfidf_ranker.score(query, top_k=num_candidates)
            combined = self._combine_rankings(bm25, tfidf, hybrid_weight)
            return combined, "Hybrid"

        if method == "dense":
            if self.dense_retriever:
                return self.dense_retriever.search(query, top_k=num_candidates), "Dense"
            return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

        if method == "hybrid_dense":
            if not self.dense_retriever:
                return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

            bm25 = self.bm25_ranker.score(query, top_k=num_candidates)
            dense = self.dense_retriever.search(query, top_k=num_candidates)
            combined = self._combine_rankings(bm25, dense, hybrid_weight)
            return combined, "Hybrid-Dense"

        if method == "splade":
            if self.splade_retriever:
                return self.splade_retriever.search(query, top_k=num_candidates), "SPLADE"
            return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

        if method == "hybrid_splade":
            if not self.splade_retriever:
                return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

            bm25 = self.bm25_ranker.score(query, top_k=num_candidates)
            splade = self.splade_retriever.search(query, top_k=num_candidates)
            combined = self._combine_rankings(bm25, splade, hybrid_weight)
            return combined, "Hybrid-Splade"

        return self.bm25_ranker.score(query, top_k=num_candidates), "BM25"

    @staticmethod
    def _normalize_scores(scores):
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score == 0:
            return {doc_id: 0.0 for doc_id in scores}
        return {doc_id: score / max_score for doc_id, score in scores.items()}

    def _combine_rankings(self, base_results, alt_results, base_weight):
        base_scores = {doc_id: score for doc_id, score in base_results}
        alt_scores = {doc_id: score for doc_id, score in alt_results}

        base_norm = self._normalize_scores(base_scores)
        alt_norm = self._normalize_scores(alt_scores)

        combined = defaultdict(float)
        for doc_id, score in base_norm.items():
            combined[doc_id] += base_weight * score
        for doc_id, score in alt_norm.items():
            combined[doc_id] += (1.0 - base_weight) * score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _extract_snippet(self, doc_id, query, max_len=200):
        """Extract a short snippet that matches the query."""
        text = self.index.get_document(doc_id)
        if not text:
            return ""

        query_terms = set(self.tokenizer.tokenize(query))
        words = text.split()

        best_start = 0
        best_count = 0
        window = 25

        for i in range(len(words)):
            count = 0
            for j in range(i, min(i + window, len(words))):
                word_clean = words[j].lower().strip('.,!?;:"\'')
                if word_clean in query_terms:
                    count += 1

            if count > best_count:
                best_count = count
                best_start = i

        start = max(0, best_start - 3)
        end = min(len(words), best_start + window)
        snippet = ' '.join(words[start:end])

        if len(snippet) > max_len:
            snippet = snippet[:max_len] + "..."

        return snippet

    def get_document(self, doc_id):
        """Return full document text."""
        return self.index.get_document(doc_id)

    def explain_doc(self, query, doc_id, method="bm25"):
        """Explain scoring for a single document and query.

        Returns a dict with per-term contributions and doc stats. For unsupported
        methods, returns an empty explanation.

        Structure:
            {
              'method': 'BM25' | 'TF-IDF' | 'N/A',
              'doc_id': str,
              'doc_len': int,
              'avg_doc_len': float,
              'terms': [
                  {
                    'term': str,
                    'tf': int,
                    'idf': float,
                    'q_weight': float | None,
                    'score': float
                  }, ...
              ],
              'total': float
            }
        """
        method = (method or "bm25").lower()
        # Only lexical methods are explainable here
        explainable = {"bm25", "tfidf", "hybrid"}
        if method not in explainable:
            return {
                "method": "N/A",
                "doc_id": doc_id,
                "doc_len": self.index.doc_len.get(doc_id, 0),
                "avg_doc_len": self.index.avg_doc_len,
                "terms": [],
                "total": 0.0,
            }

        # Use tokenization consistent with rankers
        terms_q = self.tokenizer.tokenize(query)
        if not terms_q:
            return {
                "method": method.upper(),
                "doc_id": doc_id,
                "doc_len": self.index.doc_len.get(doc_id, 0),
                "avg_doc_len": self.index.avg_doc_len,
                "terms": [],
                "total": 0.0,
            }

        # Helper: fetch tf of a term in a document
        def tf_in_doc(term, doc_id):
            posting = self.index.get_posting(term)
            for d, tf in posting:
                if d == doc_id:
                    return tf
            return 0

        # Compute BM25 components
        def bm25_idf(term):
            N = self.index.total_docs
            df = self.index.get_doc_freq(term)
            return math.log((N - df + 0.5) / (df + 0.5) + 1)

        # Compute TF-IDF components
        def tfidf_idf(term):
            total_docs = self.index.total_docs or 0
            df = self.index.get_doc_freq(term)
            return math.log((total_docs + 1) / (df + 1)) + 1.0

        def tf_log(freq):
            if freq <= 0:
                return 0.0
            return 1.0 + math.log(freq)

        import math  # local import to avoid top-level change
        doc_len = self.index.doc_len.get(doc_id, 0) or 1
        avgdl = self.index.avg_doc_len or 1.0

        # When hybrid, use BM25 explanation by default
        method_key = "bm25" if method == "hybrid" else method

        details = []
        total = 0.0

        if method_key == "bm25":
            k1 = getattr(self.bm25_ranker, "k1", 1.5)
            b = getattr(self.bm25_ranker, "b", 0.75)
            for term in terms_q:
                tf = tf_in_doc(term, doc_id)
                if tf <= 0:
                    continue
                idf = bm25_idf(term)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                score = idf * (numerator / denominator)
                total += score
                details.append({
                    "term": term,
                    "tf": tf,
                    "idf": idf,
                    "q_weight": None,
                    "score": score,
                })
            return {
                "method": "BM25",
                "doc_id": doc_id,
                "doc_len": doc_len,
                "avg_doc_len": avgdl,
                "terms": details,
                "total": total,
            }

        # TF-IDF explanation
        from collections import Counter
        q_tf = Counter(terms_q)
        q_weights = {t: tf_log(f) for t, f in q_tf.items()}
        for term, q_w in q_weights.items():
            tf = tf_in_doc(term, doc_id)
            if tf <= 0:
                continue
            idf = tfidf_idf(term)
            doc_tf = tf_log(tf)
            score = (doc_tf * idf) * q_w / doc_len
            total += score
            details.append({
                "term": term,
                "tf": tf,
                "idf": idf,
                "q_weight": q_w,
                "score": score,
            })

        return {
            "method": "TF-IDF",
            "doc_id": doc_id,
            "doc_len": doc_len,
            "avg_doc_len": avgdl,
            "terms": details,
            "total": total,
        }

