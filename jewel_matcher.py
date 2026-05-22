"""DinoV2-based jewel image matching: load model, embed images, score similarity."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
EMBEDDING_DIM = 768


@dataclass
class MatchResult:
    score: float
    verdict: str
    template_embedding: np.ndarray
    input_embedding: np.ndarray
    preprocess_ms: float
    inference_ms: float
    similarity_ms: float
    total_ms: float


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str) -> Tuple[AutoModel, AutoImageProcessor]:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return model, processor


def _preprocess(image: Image.Image, processor: AutoImageProcessor, device: str):
    rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _embed(model: AutoModel, tensor_inputs, device: str) -> np.ndarray:
    with torch.no_grad():
        outputs = model(**tensor_inputs)
    cls = outputs.last_hidden_state[:, 0, :].squeeze(0)
    cls = torch.nn.functional.normalize(cls, p=2, dim=0)
    return cls.cpu().numpy()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def embed_image(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> Tuple[np.ndarray, float, float]:
    """Return (embedding, preprocess_ms, inference_ms)."""
    t0 = time.perf_counter()
    inputs = _preprocess(image, processor, device)
    _sync(device)
    t1 = time.perf_counter()
    embedding = _embed(model, inputs, device)
    _sync(device)
    t2 = time.perf_counter()
    return embedding, (t1 - t0) * 1000, (t2 - t1) * 1000


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def match(
    template_image: Image.Image,
    input_image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    threshold: float,
) -> MatchResult:
    t_start = time.perf_counter()

    tmpl_emb, pre1, inf1 = embed_image(template_image, model, processor, device)
    inp_emb, pre2, inf2 = embed_image(input_image, model, processor, device)

    t_sim = time.perf_counter()
    score = cosine_similarity(tmpl_emb, inp_emb)
    sim_ms = (time.perf_counter() - t_sim) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    verdict = "PASS" if score >= threshold else "FAIL"

    return MatchResult(
        score=score,
        verdict=verdict,
        template_embedding=tmpl_emb,
        input_embedding=inp_emb,
        preprocess_ms=pre1 + pre2,
        inference_ms=inf1 + inf2,
        similarity_ms=sim_ms,
        total_ms=total_ms,
    )
