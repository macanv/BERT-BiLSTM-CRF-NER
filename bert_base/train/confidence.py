# encoding=utf-8
"""Utilities for computing CRF-based confidence scores."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _log_sum_exp(value: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Stable log-sum-exp implementation."""
    max_score = np.max(value, axis=axis, keepdims=True)
    stabilized = value - max_score
    sum_exp = np.sum(np.exp(stabilized), axis=axis, keepdims=True)
    result = max_score + np.log(sum_exp)
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _forward_backward_single(
    logits: np.ndarray, transition_params: np.ndarray
) -> np.ndarray:
    """Runs the forward-backward algorithm for a single sequence.

    Args:
        logits: Array with shape [seq_len, num_tags]. Each entry contains the
            unary potential for a given position and tag.
        transition_params: Array with shape [num_tags, num_tags] representing the
            transition matrix learned by the CRF layer.

    Returns:
        Array with shape [seq_len, num_tags] containing p(y_t | x) for each
        position and tag.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D array")
    if transition_params.ndim != 2:
        raise ValueError("transition_params must be a 2-D array")

    seq_len, num_tags = logits.shape
    if transition_params.shape != (num_tags, num_tags):
        raise ValueError(
            "transition_params shape %s does not match logits tag dimension %s"
            % (transition_params.shape, num_tags)
        )

    alpha = np.zeros((seq_len, num_tags), dtype=np.float64)
    beta = np.zeros((seq_len, num_tags), dtype=np.float64)

    alpha[0] = logits[0]
    for t in range(1, seq_len):
        scores = (
            alpha[t - 1][:, np.newaxis]
            + transition_params
            + logits[t][np.newaxis, :]
        )
        alpha[t] = _log_sum_exp(scores, axis=0)

    beta[-1] = 0.0
    for t in range(seq_len - 2, -1, -1):
        scores = (
            transition_params
            + logits[t + 1][np.newaxis, :]
            + beta[t + 1][np.newaxis, :]
        )
        beta[t] = _log_sum_exp(scores, axis=1)

    log_z = _log_sum_exp(alpha[-1], axis=0)
    log_marginals = alpha + beta - log_z
    marginals = np.exp(log_marginals)
    return np.clip(marginals, 0.0, 1.0)


def compute_token_marginals(
    logits: np.ndarray,
    transition_params: np.ndarray,
    sequence_lengths: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    """Computes CRF token-level marginals for a batch of sequences.

    Args:
        logits: A NumPy array with shape [batch_size, max_seq_len, num_tags] or
            [max_seq_len, num_tags] if a single sequence is provided.
        transition_params: The CRF transition matrix with shape [num_tags,
            num_tags].
        sequence_lengths: Optional iterable specifying the true sequence length
            for each batch element. If omitted, the full length of each sequence
            is used.

    Returns:
        A list of arrays. Each array has shape [seq_len, num_tags] containing the
        marginal distribution over tags for every position in the sequence.
    """
    logits = np.asarray(logits)
    transition_params = np.asarray(transition_params)

    if logits.ndim == 2:
        if sequence_lengths is None:
            used = logits.shape[0]
        else:
            try:
                used = int(sequence_lengths)
            except TypeError:
                raise ValueError(
                    "For 2-D logits, sequence_lengths must be a scalar or None"
                )
        return [_forward_backward_single(logits[:used], transition_params)]

    if logits.ndim != 3:
        raise ValueError("logits must be a 2-D or 3-D array")

    batch_size, max_seq_len, _ = logits.shape
    if sequence_lengths is None:
        sequence_lengths = [max_seq_len] * batch_size
    if len(sequence_lengths) != batch_size:
        raise ValueError(
            "Expected %d sequence lengths but got %d"
            % (batch_size, len(sequence_lengths))
        )

    marginals = []
    for seq_logits, seq_len in zip(logits, sequence_lengths):
        used = int(seq_len)
        if used <= 0:
            marginals.append(np.zeros((0, logits.shape[-1]), dtype=np.float64))
            continue
        marginals.append(
            _forward_backward_single(seq_logits[:used], transition_params)
        )
    return marginals


def _iter_entity_spans(tags: Sequence[str]) -> Iterable[Tuple[int, int, str]]:
    """Yields (start, end, entity_type) tuples for a tag sequence."""
    start = None
    entity_type = None

    for index, tag in enumerate(tags):
        if tag is None:
            continue
        if tag == "O" or tag.startswith("["):
            if start is not None:
                yield start, index - 1, entity_type
                start = None
                entity_type = None
            continue
        if "-" in tag:
            prefix, type_name = tag.split("-", 1)
        else:
            prefix, type_name = tag, ""

        if prefix == "B" or prefix == "S":
            if start is not None:
                yield start, index - 1, entity_type
            start = index
            entity_type = type_name
            if prefix == "S":
                yield index, index, type_name
                start = None
                entity_type = None
        elif prefix in ("I", "E"):
            if start is None:
                start = index
                entity_type = type_name
            if prefix == "E":
                yield start, index, entity_type
                start = None
                entity_type = None
        else:
            # Unknown prefix, close any open span.
            if start is not None:
                yield start, index - 1, entity_type
            start = None
            entity_type = None

    if start is not None:
        yield start, len(tags) - 1, entity_type


def aggregate_span_confidence(
    tags: Sequence[str],
    token_marginals: np.ndarray,
    label_to_id: Dict[str, int],
    tokens: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Aggregates token marginals into entity-level confidence scores.

    Args:
        tags: Sequence of predicted tags (e.g. BIO or BIES format) aligned with
            the token marginals.
        token_marginals: Array with shape [seq_len, num_tags] returned by
            :func:`compute_token_marginals`.
        label_to_id: Mapping from tag string to its CRF label index.
        tokens: Optional original tokens aligned with ``tags``. When provided,
            they are attached to the returned span metadata.

    Returns:
        A list of dictionaries, one per detected entity span. Each dictionary
        contains ``start`` (inclusive index), ``end`` (inclusive index), ``type``
        (entity type string) and three aggregated confidence scores:
        ``min_prob``, ``mean_prob`` and ``prod_prob``.
    """
    if token_marginals.ndim != 2:
        raise ValueError("token_marginals must be a 2-D array")

    results: List[Dict[str, object]] = []
    for start, end, entity_type in _iter_entity_spans(tags):
        probs: List[float] = []
        for position in range(start, end + 1):
            tag = tags[position]
            if tag not in label_to_id:
                raise KeyError("Tag %r not found in label_to_id mapping" % tag)
            probs.append(float(token_marginals[position, label_to_id[tag]]))

        if not probs:
            continue

        clipped = np.clip(probs, 1e-12, 1.0)
        span_info: Dict[str, object] = {
            "start": start,
            "end": end,
            "type": entity_type,
            "length": end - start + 1,
            "min_prob": float(np.min(clipped)),
            "mean_prob": float(np.mean(clipped)),
            "prod_prob": float(np.exp(np.sum(np.log(clipped)))),
            "token_probs": probs,
        }
        if tokens is not None:
            span_info["tokens"] = list(tokens[start : end + 1])
        results.append(span_info)
    return results
