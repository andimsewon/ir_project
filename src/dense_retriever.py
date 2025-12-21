"""
Dense retriever module using HuggingFace sentence-transformers.
Builds and searches a document embedding index.
"""
from typing import Dict, List, Tuple, Union

import importlib.util

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseRetriever:
    """
    Dense retriever with a bi-encoder model.

    Stores normalized document embeddings for cosine similarity search.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: Union[str, torch.device] = None, max_length: int = 512):
        self.device = _resolve_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if max_length:
            try:
                self.model.max_seq_length = max_length
            except Exception:
                pass

        self.doc_ids: List[str] = []
        self.embeddings: torch.Tensor = None

    def build_index(self, doc_store: Dict[str, str], batch_size: int = 64, show_progress: bool = True):
        """Encode all documents and store normalized embeddings."""
        items = list(doc_store.items())
        self.doc_ids = []
        all_embeddings = []

        iterator = range(0, len(items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents")

        for start in iterator:
            batch = items[start:start + batch_size]
            batch_ids = [doc_id for doc_id, _ in batch]
            batch_texts = [text for _, text in batch]

            batch_emb = self._encode_texts(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=True
            )
            batch_emb = batch_emb.to("cpu")

            self.doc_ids.extend(batch_ids)
            all_embeddings.append(batch_emb)

        if all_embeddings:
            self.embeddings = torch.cat(all_embeddings, dim=0)
        else:
            self.embeddings = None

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Return top_k (doc_id, score) pairs using cosine similarity."""
        if self.embeddings is None or not self.doc_ids:
            return []

        query_emb = self._encode_texts(
            query,
            normalize_embeddings=True
        ).to(self.embeddings.device)

        scores = torch.matmul(self.embeddings, query_emb)
        if scores.dim() > 1:
            scores = scores.squeeze(-1)

        k = min(top_k, scores.numel())
        top_scores, top_idx = torch.topk(scores, k=k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            results.append((self.doc_ids[idx], float(score)))

        return results

    def _encode_texts(
        self,
        texts,
        batch_size: int = 32,
        normalize_embeddings: bool = False
    ) -> torch.Tensor:
        encode_wrapped = self.model.encode
        if hasattr(encode_wrapped, "__wrapped__") and getattr(self.device, "type", None) == "privateuseone":
            return encode_wrapped.__wrapped__(
                self.model,
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings
            )

        return encode_wrapped(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings
        )

    def save(self, path: str):
        """Save embeddings and ids to disk."""
        data = {
            "model_name": self.model_name,
            "doc_ids": self.doc_ids,
            "embeddings": self.embeddings,
        }
        torch.save(data, path)

    def load(self, path: str):
        """Load embeddings and ids from disk."""
        data = torch.load(path, map_location="cpu")
        self.model_name = data["model_name"]
        self.doc_ids = data["doc_ids"]
        self.embeddings = data["embeddings"]

        if self.model is None or self.model_name != getattr(self.model, "name_or_path", None):
            self.model = SentenceTransformer(self.model_name, device=self.device)


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
