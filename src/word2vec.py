"""NumPy implementation of Word2Vec (Skip-gram and CBOW) with negative sampling for Homework 2.

Learns center and context matrices from local sentences, using unigram^0.75
noise and optional frequent-word subsampling. No pretrained vectors are used.
Skip-gram: each word predicts every word in its window. CBOW: the mean of the
window's vectors predicts the center word (original word2vec cbow_mean=1).
Reference: Mikolov et al. (2013), https://arxiv.org/abs/1310.4546
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

MODEL_VERSION = "numpy-sgns-v1"
ARCHITECTURES = {"skipgram": "Skip-gram with negative sampling", "cbow": "CBOW with negative sampling"}


@dataclass(frozen=True)
class Word2VecConfig:
    dimensions: int = 64
    window: int = 5
    min_count: int = 2
    negatives: int = 5
    epochs: int = 15
    learning_rate: float = 0.025
    sample: float = 0.001
    batch_size: int = 128
    seed: int = 42
    architecture: str = "skipgram"  # "skipgram": word -> context; "cbow": averaged context -> word

    @classmethod
    def from_dict(cls, settings: dict) -> "Word2VecConfig":
        """Rebuild a config from saved metadata, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in settings.items() if k in known})

    def validate(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {', '.join(ARCHITECTURES)}.")
        for name, lower, upper in (("dimensions", 2, 300), ("window", 1, 15), ("min_count", 1, 100),
                                   ("negatives", 1, 20), ("epochs", 1, 100), ("batch_size", 1, 512)):
            value = getattr(self, name)
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}.")
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be in (0, 0.1].")
        if not np.isfinite(self.sample) or not 0 <= self.sample <= 1:
            raise ValueError("sample must be in [0, 1]; zero disables subsampling.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer.")


def model_key(config: Word2VecConfig, corpus_fingerprint: str, preprocessing: dict) -> str:
    settings = asdict(config)
    if settings["architecture"] == "skipgram":
        del settings["architecture"]  # keeps keys of models trained before CBOW existed
    payload = {"version": MODEL_VERSION, "config": settings,
               "corpus": corpus_fingerprint, "preprocessing": preprocessing}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class Word2VecModel:
    words: list[str]
    counts: np.ndarray
    vectors: np.ndarray
    context_vectors: np.ndarray
    metadata: dict = field(default_factory=dict)

    def most_similar(self, word: str, top_n: int = 10) -> list[dict]:
        if word not in self.words:
            raise ValueError(f"'{word}' is not in the trained vocabulary. Try another word or lower min_count.")
        index = self.words.index(word)
        lengths = np.linalg.norm(self.vectors, axis=1)
        normalized = self.vectors / np.maximum(lengths[:, None], 1e-12)
        scores = normalized @ normalized[index]
        ordered = np.argsort(-scores, kind="stable")
        return [{"word": self.words[i], "cosine_similarity": float(scores[i]), "corpus_count": int(self.counts[i])}
                for i in ordered if i != index][:max(0, top_n)]

    def to_bytes(self) -> bytes:
        output = io.BytesIO()
        np.savez_compressed(output, words=np.array(self.words, dtype=str), counts=self.counts,
                            vectors=self.vectors, context_vectors=self.context_vectors,
                            metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)))
        return output.getvalue()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after the full archive has been written.
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(self.to_bytes())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "Word2VecModel":
        with np.load(path, allow_pickle=False) as archive:
            result = cls(archive["words"].tolist(), archive["counts"], archive["vectors"],
                         archive["context_vectors"], json.loads(str(archive["metadata"])))
        if (result.vectors.ndim != 2 or result.context_vectors.shape != result.vectors.shape
                or len(result.words) != len(set(result.words)) or len(result.words) != len(result.vectors)
                or result.counts.shape != (len(result.words),)
                or not np.isfinite(result.vectors).all() or not np.isfinite(result.context_vectors).all()
                or result.metadata.get("version") != MODEL_VERSION):
            raise ValueError("Invalid or incompatible Word2Vec model.")
        return result


def skipgram_pairs(sentences: list[np.ndarray], window: int) -> np.ndarray:
    """Directed center-context pairs, without crossing sentence boundaries."""
    chunks = []
    for sentence in sentences:
        for offset in range(1, min(window + 1, len(sentence))):
            chunks.append(np.column_stack((sentence[:-offset], sentence[offset:])))
            chunks.append(np.column_stack((sentence[offset:], sentence[:-offset])))
    return np.concatenate(chunks).astype(np.int32) if chunks else np.empty((0, 2), dtype=np.int32)


def context_window(words: list[str], center: int, window: int) -> list[dict]:
    """Skip-gram training examples produced by one center position (for the demo)."""
    if not 0 <= center < len(words):
        raise ValueError("Center position is outside the sentence.")
    return [{"center": words[center], "context": words[position], "offset": position - center, "position": position}
            for position in range(max(0, center - window), min(len(words), center + window + 1)) if position != center]


def pair_scores(model: "Word2VecModel", center: str, contexts: list[str]) -> list[dict]:
    """sigma(v_center . u_context): the trained probability that a pair is a true pair."""
    if center not in model.words:
        raise ValueError(f"'{center}' is not in the trained vocabulary.")
    vector = model.vectors[model.words.index(center)]
    rows = []
    for word in contexts:
        if word in model.words:
            score = float(vector @ model.context_vectors[model.words.index(word)])
            rows.append({"context": word, "dot_product": score, "sigmoid": float(1 / (1 + np.exp(-score)))})
    return rows


def _noise(rng: np.random.Generator, targets: np.ndarray, probabilities: np.ndarray, count: int) -> np.ndarray:
    # Draw with replacement from unigram^0.75. A draw equal to the positive
    # context is masked in the objective (the original word2vec convention).
    return rng.choice(len(probabilities), size=(len(targets), count), p=probabilities)


def cbow_examples(sentence: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """CBOW training examples: for every position, its window (padded with -1) and the center word."""
    n = len(sentence)
    if n < 2:
        return np.empty((0, 2 * window), dtype=np.int32), np.empty(0, dtype=np.int32)
    contexts = np.full((n, 2 * window), -1, dtype=np.int32)
    for column, offset in enumerate([o for o in range(-window, window + 1) if o]):
        positions = np.arange(n) + offset
        valid = (positions >= 0) & (positions < n)
        contexts[valid, column] = sentence[positions[valid]]
    return contexts, sentence.astype(np.int32)


def _context_mean(inputs: np.ndarray, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = contexts >= 0
    sizes = mask.sum(axis=1, keepdims=True).astype(np.float32)
    hidden = (inputs[np.where(mask, contexts, 0)] * mask[:, :, None]).sum(axis=1) / sizes
    return hidden, sizes


def _loss_vectors(hidden: np.ndarray, outputs: np.ndarray, targets: np.ndarray, noise: np.ndarray) -> float:
    positives = np.einsum("ij,ij->i", hidden, outputs[targets])
    negative_scores = np.einsum("ij,ikj->ik", hidden, outputs[noise])
    mask = noise != targets[:, None]
    return float(np.mean(np.logaddexp(0, -positives) + (np.logaddexp(0, negative_scores) * mask).sum(axis=1)))


def _loss(inputs: np.ndarray, outputs: np.ndarray, pairs: np.ndarray, noise: np.ndarray) -> float:
    centers = inputs[pairs[:, 0]]
    positives = np.einsum("ij,ij->i", centers, outputs[pairs[:, 1]])
    negative_scores = np.einsum("ij,ikj->ik", centers, outputs[noise])
    mask = noise != pairs[:, 1, None]
    return float(np.mean(np.logaddexp(0, -positives) + (np.logaddexp(0, negative_scores) * mask).sum(axis=1)))


def _cbow_sentence(sentence, config, rng, inputs, outputs, probabilities, rate) -> int:
    """One sentence of CBOW: the mean of the window's input vectors predicts the center word."""
    contexts, centers = cbow_examples(sentence, config.window)
    order = rng.permutation(len(centers))
    contexts, centers = contexts[order], centers[order]
    for offset in range(0, len(centers), config.batch_size):
        batch_contexts, center_ids = contexts[offset:offset + config.batch_size], centers[offset:offset + config.batch_size]
        negative_ids = _noise(rng, center_ids, probabilities, config.negatives)
        targets = np.column_stack((center_ids, negative_ids))
        hidden, _ = _context_mean(inputs, batch_contexts)
        target_vectors = outputs[targets]
        scores = np.einsum("ij,ikj->ik", hidden, target_vectors)
        gradient = 1 / (1 + np.exp(-np.clip(scores, -20, 20)))
        gradient[:, 0] -= 1
        gradient[:, 1:] *= negative_ids != center_ids[:, None]
        # As in the original word2vec (cbow_mean=1): the hidden vector is the context mean,
        # and its full gradient is applied to every context word.
        hidden_gradient = np.einsum("ik,ikj->ij", gradient, target_vectors)
        mask = batch_contexts >= 0
        np.add.at(inputs, batch_contexts[mask], -rate * np.repeat(hidden_gradient, mask.sum(axis=1), axis=0))
        np.add.at(outputs, targets.ravel(), -rate * (gradient[:, :, None] * hidden[:, None, :]).reshape(-1, config.dimensions))
    return len(centers)


def train_skipgram(
    sentences: list[list[str]], config: Word2VecConfig, *, corpus_metadata: dict | None = None,
    preprocessing: dict | None = None, progress: Callable[[int, int, float], None] | None = None,
) -> Word2VecModel:
    config.validate()
    started = perf_counter()
    frequencies = Counter(word for sentence in sentences for word in sentence)
    words = sorted((word for word, count in frequencies.items() if count >= config.min_count),
                   key=lambda word: (-frequencies[word], word))
    if len(words) < 2:
        raise ValueError("Training needs at least two vocabulary words. Lower min_count or use more documents.")
    by_word = {word: index for index, word in enumerate(words)}
    counts = np.array([frequencies[word] for word in words], dtype=np.int64)
    encoded = [np.array([by_word[word] for word in sentence if word in by_word], dtype=np.int32)
               for sentence in sentences]
    encoded = [sentence for sentence in encoded if len(sentence) >= 2]
    if not encoded:
        raise ValueError("No center-context pairs remain after vocabulary filtering.")
    rng = np.random.default_rng(config.seed)
    inputs = rng.uniform(-0.5 / config.dimensions, 0.5 / config.dimensions,
                         (len(words), config.dimensions)).astype(np.float32)
    outputs = np.zeros_like(inputs)
    probabilities = counts.astype(float) ** 0.75
    probabilities /= probabilities.sum()
    relative = counts / counts.sum()
    keep = np.minimum(1, (np.sqrt(relative / config.sample) + 1) * config.sample / relative) if config.sample else np.ones(len(words))

    cbow = config.architecture == "cbow"
    # Fixed training probe, not a validation set or an accuracy estimate.
    # Construct at most 2048 pairs, so a 1000-document corpus stays bounded.
    probe_chunks = []
    probe_size = 0
    for i in rng.permutation(len(encoded)):
        pairs = (np.column_stack(cbow_examples(encoded[i], config.window)) if cbow
                 else skipgram_pairs([encoded[i]], config.window))
        take = pairs[:2048 - probe_size]
        probe_chunks.append(take)
        probe_size += len(take)
        if probe_size >= 2048:
            break
    probe = np.concatenate(probe_chunks)
    probe_noise = _noise(rng, probe[:, -1], probabilities, config.negatives)

    def probe_loss():
        if cbow:  # rows are [context ids (padded -1) ..., center id]
            return _loss_vectors(_context_mean(inputs, probe[:, :-1])[0], outputs, probe[:, -1], probe_noise)
        return _loss(inputs, outputs, probe, probe_noise)

    history = [{"epoch": 0, "probe_loss": probe_loss(), "pairs": 0}]
    pairs_total = 0
    for epoch in range(1, config.epochs + 1):
        # Shuffle documents/sentences and local pairs rather than allocating all
        # pairs for the corpus at once. Context windows never cross a boundary.
        pair_count = 0
        rate = config.learning_rate * (1 - 0.9 * (epoch - 1) / max(config.epochs - 1, 1))
        for sentence_id in rng.permutation(len(encoded)):
            sentence = encoded[sentence_id]
            sentence = sentence[rng.random(len(sentence)) < keep[sentence]]
            if cbow:
                pair_count += _cbow_sentence(sentence, config, rng, inputs, outputs, probabilities, rate)
                continue
            pairs = skipgram_pairs([sentence], config.window)
            rng.shuffle(pairs)
            for offset in range(0, len(pairs), config.batch_size):
                batch = pairs[offset:offset + config.batch_size]
                center_ids, target_ids = batch[:, 0], batch[:, 1]
                negative_ids = _noise(rng, target_ids, probabilities, config.negatives)
                targets = np.column_stack((target_ids, negative_ids))
                # Advanced indexing produces copies: both gradients use the
                # same pre-update weights, including when IDs repeat.
                center_vectors = inputs[center_ids]
                target_vectors = outputs[targets]
                scores = np.einsum("ij,ikj->ik", center_vectors, target_vectors)
                gradient = 1 / (1 + np.exp(-np.clip(scores, -20, 20)))
                gradient[:, 0] -= 1
                gradient[:, 1:] *= negative_ids != target_ids[:, None]
                center_gradient = np.einsum("ik,ikj->ij", gradient, target_vectors)
                context_gradient = gradient[:, :, None] * center_vectors[:, None, :]
                # Accumulated SGD updates; add.at correctly handles repeated
                # center/context IDs within one small batch.
                np.add.at(inputs, center_ids, -rate * center_gradient)
                np.add.at(outputs, targets.ravel(), -rate * context_gradient.reshape(-1, config.dimensions))
            pair_count += len(pairs)
        if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
            raise ValueError("Training diverged. Use a smaller learning rate or batch size.")
        pairs_total += pair_count
        loss = probe_loss()
        history.append({"epoch": epoch, "probe_loss": loss, "pairs": pair_count})
        if progress:
            progress(epoch, config.epochs, loss)
    if not pairs_total:
        raise ValueError("Subsampling removed all context pairs. Disable subsampling or add more documents.")
    corpus_metadata = corpus_metadata or {}
    preprocessing = preprocessing or {}
    metadata = {"version": MODEL_VERSION, "architecture": ARCHITECTURES[config.architecture],
                "config": asdict(config), "corpus": corpus_metadata, "preprocessing": preprocessing,
                "model_key": model_key(config, corpus_metadata.get("fingerprint", ""), preprocessing),
                "vocabulary_size": len(words), "raw_tokens": sum(frequencies.values()),
                "retained_tokens": int(counts.sum()), "sentence_count": len(encoded),
                "training_pairs": pairs_total, "history": history,
                "elapsed_seconds": perf_counter() - started,
                "loss_note": "Mean negative-sampling loss on fixed training pairs/noise, not held-out accuracy."}
    return Word2VecModel(words, counts, inputs, outputs, metadata)
