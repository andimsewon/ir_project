"""
Dense retriever module using HuggingFace sentence-transformers.
Builds and searches a document embedding index.
"""
from typing import Dict, List, Tuple, Union

import importlib.util
import os

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
except Exception:
    faiss = None


class DenseRetriever:
    """
    Dense retriever with a bi-encoder model.

    Stores normalized document embeddings for cosine similarity search.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Union[str, torch.device] = None,
        max_length: int = 512,
        ann_enabled: bool = True,
        ann_m: int = 32,
        ann_ef_construction: int = 200,
        ann_ef_search: int = 128,
    ):
        self.device = _resolve_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.ann_available = faiss is not None
        self.ann_enabled = ann_enabled and self.ann_available
        self.ann_m = ann_m
        self.ann_ef_construction = ann_ef_construction
        self.ann_ef_search = ann_ef_search

        if max_length:
            try:
                self.model.max_seq_length = max_length
            except Exception:
                pass

        self.doc_ids: List[str] = []
        self.embeddings: torch.Tensor = None
        self.ann_index = None

    def build_index(self, doc_store: Dict[str, str], batch_size: int = 64, show_progress: bool = True):
        # encode docs in batches
        """Encode all documents and store normalized embeddings."""
        items = list(doc_store.items())
        self.doc_ids = []
        all_embeddings = []

        current_batch = max(1, batch_size)
        idx = 0
        pbar = None
        if show_progress:
            pbar = tqdm(total=len(items), desc="Encoding documents")

        while idx < len(items):
            batch = items[idx:idx + current_batch]
            batch_ids = [doc_id for doc_id, _ in batch]
            batch_texts = [text for _, text in batch]

            try:
                batch_emb = self._encode_texts(
                    batch_texts,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True
                )
            except RuntimeError as exc:
                if _is_oom_error(exc) and current_batch > 1 and str(self.device).lower() != "cpu":
                    current_batch = max(1, current_batch // 2)
                    _clear_device_cache()
                    if pbar:
                        pbar.set_postfix_str(f"OOM, batch -> {current_batch}")
                    continue
                raise

            batch_emb = batch_emb.to("cpu")

            self.doc_ids.extend(batch_ids)
            all_embeddings.append(batch_emb)
            idx += len(batch)
            if pbar:
                pbar.update(len(batch))

        if pbar:
            pbar.close()

        if all_embeddings:
            self.embeddings = torch.cat(all_embeddings, dim=0)
        else:
            self.embeddings = None

        self._build_ann_index()

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Return top_k (doc_id, score) pairs using cosine similarity."""
        if self.embeddings is None or not self.doc_ids:
            return []

        if self.ann_index is not None:
            query_emb = self._encode_texts(
                query,
                normalize_embeddings=True
            ).to("cpu")
            query_vec = query_emb.numpy().astype("float32")
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)

            k = min(top_k, len(self.doc_ids))
            distances, indices = self.ann_index.search(query_vec, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                score = 1.0 - (float(dist) / 2.0)
                results.append((self.doc_ids[idx], score))
            return results

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

        # build FAISS ANN index if enabled
def _build_ann_index(self):
        if not self.ann_enabled or self.embeddings is None:
            self.ann_index = None
            return

        embeddings = self.embeddings
        if embeddings.device.type != "cpu":
            embeddings = embeddings.to("cpu")

        emb_np = embeddings.numpy().astype("float32")
        dim = emb_np.shape[1]

        index = faiss.IndexHNSWFlat(dim, self.ann_m)
        index.hnsw.efConstruction = self.ann_ef_construction
        index.hnsw.efSearch = self.ann_ef_search
        index.add(emb_np)
        self.ann_index = index

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
            "ann_enabled": self.ann_enabled,
            "ann_m": self.ann_m,
            "ann_ef_construction": self.ann_ef_construction,
            "ann_ef_search": self.ann_ef_search,
        }
        torch.save(data, path)
        if self.ann_index is not None:
            faiss.write_index(self.ann_index, _faiss_path(path))

    def load(self, path: str):
        """Load embeddings and ids from disk."""
        data = torch.load(path, map_location="cpu")
        self.model_name = data["model_name"]
        self.doc_ids = data["doc_ids"]
        self.embeddings = data["embeddings"]
        self.ann_enabled = data.get("ann_enabled", False) and self.ann_available
        self.ann_m = data.get("ann_m", self.ann_m)
        self.ann_ef_construction = data.get("ann_ef_construction", self.ann_ef_construction)
        self.ann_ef_search = data.get("ann_ef_search", self.ann_ef_search)
        self.ann_index = None

        if self.ann_enabled:
            faiss_path = _faiss_path(path)
            if os.path.exists(faiss_path):
                self.ann_index = faiss.read_index(faiss_path)
                self.ann_index.hnsw.efSearch = self.ann_ef_search

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


def _faiss_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.faiss"


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "not enough gpu video memory" in msg
        or "could not allocate tensor" in msg
    )


def _clear_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
