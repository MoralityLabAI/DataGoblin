"""A shared reasoning and answer cell for the mixed-task pretraining pilot."""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np

from .rlt_pilot import feedback_weight

ARMS = {
    "direct": ("direct", "direct"),
    "answer_rlt": ("direct", "rlt"),
    "answer_damping": ("direct", "matched_damping"),
    "reasoning_rlt": ("rlt", "direct"),
    "reasoning_damping": ("matched_damping", "direct"),
}
VOCAB = 258
ROUTE_FEATURES = 3


def initialize(seed, width, lean_features):
    rng = np.random.default_rng(seed)
    scale = width**-0.5
    return {
        "token_embedding": rng.normal(0, 0.20, (VOCAB, width)),
        "lean_input": rng.normal(0, lean_features**-0.5, (lean_features, width)),
        "route_input": rng.normal(0, ROUTE_FEATURES**-0.5, (ROUTE_FEATURES, width)),
        "self": rng.normal(0, 0.35 * scale, (width, width)),
        "message": rng.normal(0, 0.45 * scale, (width, width)),
        "partner": rng.normal(0, 0.20 * scale, (width, width)),
        "phase": np.zeros((2, width)),
        "proposal": rng.normal(0, scale, (2, width)),
        "clock": rng.normal(0, 0.10, (3, width)),
    }


def mix(state, endpoint, mode, visit, depth):
    eta = feedback_weight(mode, visit, depth)
    return endpoint if eta == 1.0 else state + eta * (endpoint - state)


def recurrent_endpoint(weights, encoded, state, message, partner, phase):
    return anp.tanh(
        encoded + anp.dot(state, weights["self"]) + anp.dot(message, weights["message"])
        + anp.dot(partner, weights["partner"]) + weights["phase"][phase]
    )


def refine(weights, encoded, links, mask, arm, depth, *, prior_answer=None, stop_when=None):
    """Yield actual visited states. Interpolation keeps the planned segment clock."""
    reasoning, answer = ARMS[arm]
    if depth < 1:
        raise ValueError("planned depth must be positive")
    present = mask[..., None]
    latent = anp.zeros(encoded.shape)
    carry = anp.zeros(encoded.shape) if prior_answer is None else prior_answer * present
    for index in range(depth):
        clock = anp.array([index / depth, 1 / depth, 1 - index / depth])
        context = encoded + anp.dot(clock, weights["clock"])
        routed = anp.matmul(links, latent)
        candidate = recurrent_endpoint(weights, context, latent, routed, carry, 0)
        latent = mix(latent, candidate, reasoning, index, depth) * present
        revision = recurrent_endpoint(weights, context, carry, latent, latent, 1)
        carry = mix(carry, revision, answer, index, depth) * present
        yield latent, carry
        if stop_when is not None and stop_when(index + 1, carry):
            break


def pooled_links(mask):
    weights = mask[:, None, :] * mask[:, :, None]
    return weights / np.maximum(1, weights.sum(axis=-1, keepdims=True))


def causal_links(mask):
    edges = np.tril(np.ones((mask.shape[1], mask.shape[1])), k=-1)
    weights = edges[None, :, :] * mask[:, None, :] * mask[:, :, None]
    return weights / np.maximum(1, weights.sum(axis=-1, keepdims=True))


def text_visits(weights, tokens, present, arm, depth):
    # Current tokens plus strictly earlier messages predict the next token.
    encoded = weights["token_embedding"][tokens]
    links = causal_links(present)
    for _, answer in refine(weights, encoded, links, present, arm, depth):
        yield anp.dot(answer, weights["token_embedding"].T)


def lean_visits(weights, features, present, arm, depth):
    encoded = anp.dot(features, weights["lean_input"])
    links = pooled_links(present)
    for _, answer in refine(weights, encoded, links, present, arm, depth):
        yield anp.dot(answer, weights["proposal"][0])


def route_visits(weights, features, edges, arm, depth, prior_answer=None):
    encoded = anp.dot(features, weights["route_input"])
    present = np.ones(features.shape[:2], dtype=bool)
    for _, answer in refine(weights, encoded, edges, present, arm, depth, prior_answer=prior_answer):
        yield anp.dot(answer, weights["proposal"][1]), answer
