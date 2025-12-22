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

