"""
SPLADE retriever module using HuggingFace transformers.
Builds and searches a sparse inverted index from SPLADE token weights.
"""
from typing import Dict, List, Tuple, Union
import importlib.util
import math

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm


class SpladeRetriever:
    """
    SPLADE retriever with a masked language model.

    Stores a sparse inverted index of token weights per document.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: Union[str, torch.device, None] = None,
        max_length: int = 256,
        top_terms: int = 128,
        query_top_terms: int = 32,
    ):
        self.device = _resolve_device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.top_terms = top_terms
        self.query_top_terms = query_top_terms

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, use_safetensors=True)
        except Exception:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.doc_ids: List[str] = []
        self.doc_norms: List[float] = []
        self.postings: Dict[int, List[Tuple[int, float]]] = {}

    def build_index(self, doc_store: Dict[str, str], batch_size: int = 8, show_progress: bool = True):
        """Encode all documents and store sparse postings."""
        items = list(doc_store.items())
        self.doc_ids = []
        self.doc_norms = []
        self.postings = {}

        iterator = range(0, len(items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents (SPLADE)")

        for start in iterator:
            batch = items[start:start + batch_size]
            batch_ids = [doc_id for doc_id, _ in batch]
            batch_texts = [text for _, text in batch]

            doc_vecs = self._encode_texts(batch_texts)

            for doc_id, vec in zip(batch_ids, doc_vecs):
                doc_idx = len(self.doc_ids)
                self.doc_ids.append(doc_id)

                if vec.numel() == 0:
                    self.doc_norms.append(1.0)
                    continue

                top_vals, top_idx = self._topk_sparse(vec, self.top_terms)
                norm = math.sqrt(sum(v * v for v in top_vals)) or 1.0
                self.doc_norms.append(norm)

                for token_id, weight in zip(top_idx, top_vals):
                    postings = self.postings.setdefault(int(token_id), [])
                    postings.append((doc_idx, float(weight)))

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Return top_k (doc_id, score) pairs using sparse dot product."""
        if not self.postings or not self.doc_ids:
            return []

        query_vec = self._encode_texts([query])[0]
        if query_vec.numel() == 0:
            return []

        q_vals, q_idx = self._topk_sparse(query_vec, self.query_top_terms)
        q_norm = math.sqrt(sum(v * v for v in q_vals)) or 1.0

        scores: Dict[int, float] = {}
        for token_id, q_weight in zip(q_idx, q_vals):
            postings = self.postings.get(int(token_id), [])
            for doc_idx, d_weight in postings:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + (q_weight * d_weight)

        results = []
        for doc_idx, score in scores.items():
            denom = self.doc_norms[doc_idx] * q_norm
            results.append((self.doc_ids[doc_idx], score / denom if denom else score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, path: str):
        """Save SPLADE index to disk."""
        data = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "top_terms": self.top_terms,
            "query_top_terms": self.query_top_terms,
            "doc_ids": self.doc_ids,
            "doc_norms": self.doc_norms,
            "postings": self.postings,
        }
        torch.save(data, path)

    def load(self, path: str):
        """Load SPLADE index from disk."""
        data = torch.load(path, map_location="cpu")
        self.model_name = data["model_name"]
        self.max_length = data.get("max_length", self.max_length)
        self.top_terms = data.get("top_terms", self.top_terms)
        self.query_top_terms = data.get("query_top_terms", self.query_top_terms)
        self.doc_ids = data["doc_ids"]
        self.doc_norms = data["doc_norms"]
        self.postings = data["postings"]

        if self.model is None or self.model_name != getattr(self.model, "name_or_path", None):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, use_safetensors=True)
            except Exception:
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def _encode_texts(self, texts: List[str]) -> List[torch.Tensor]:
        # compute max-pooled token weights
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            weights = torch.log1p(torch.relu(logits))
            vecs = weights.max(dim=1).values

        return [vec.cpu() for vec in vecs]

    @staticmethod
    def _topk_sparse(vec: torch.Tensor, k: int) -> Tuple[List[float], List[int]]:
        if vec.numel() == 0:
            return [], []
        k = min(k, vec.numel())
        top_vals, top_idx = torch.topk(vec, k=k)
        return top_vals.tolist(), top_idx.tolist()


def _pick_device() -> Union[str, torch.device]:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    if importlib.util.find_spec("torch_directml") is not None:
        import torch_directml
        return torch_directml.device()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _resolve_device(device: Union[str, torch.device, None]) -> Union[str, torch.device]:
    if device is None:
        return _pick_device()

    if not isinstance(device, str):
        return device

    device_key = device.lower()
    if device_key in ("dml", "directml"):
        if importlib.util.find_spec("torch_directml") is None:
            raise RuntimeError(
                "Device 'dml' requested but torch-directml is not installed. "
                "Install it with: pip install torch-directml"
            )
        import torch_directml
        return torch_directml.device()

    return device_key
