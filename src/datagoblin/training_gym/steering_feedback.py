"""Versioned feedback selection for bounded mechanistic training windows."""
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Cost:
    wall_ns: int
    cpu_ns: int
    gpu_ns: int
    visits: int
    verifier_calls: int
    ram_bytes: int
    vram_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in self.vector()):
            raise ValueError('cost coordinates must be nonnegative integers')

    def vector(self):
        return (self.wall_ns, self.cpu_ns, self.gpu_ns, self.visits, self.verifier_calls,
                self.ram_bytes, self.vram_bytes)

    def fits(self, caps):
        return all(a <= b for a, b in zip(self.vector(), caps.vector(), strict=True))


@dataclass(frozen=True)
class Action:
    key: str
    site: str
    direction: str
    dose: float

    def __post_init__(self):
        if not all((self.key, self.site, self.direction)) or not isfinite(self.dose) or self.dose <= 0:
            raise ValueError('invalid action')


@dataclass(frozen=True)
class Audit:
    action: Action
    checkpoint: str
    feedback_cohort: str
    step: int
    scope: str
    specificity: float
    ideal: float
    gain_lower: float | None
    window_cost_upper: Cost | None

    def __post_init__(self):
        if not self.checkpoint or not self.feedback_cohort or type(self.step) is not int or self.step < 0:
            raise ValueError('invalid audit binding')
        if not isfinite(self.specificity) or not -1 <= self.specificity <= 1:
            raise ValueError('invalid paired specificity')
        if not isfinite(self.ideal) or not 0 < self.ideal <= 1:
            raise ValueError('invalid ideal response')
        if self.gain_lower is not None and not isfinite(self.gain_lower):
            raise ValueError('invalid gain bound')


@dataclass(frozen=True)
class Decision:
    action: Action | None
    checkpoint: str
    feedback_cohort: str
    start_step: int
    stop_step: int
    scores: tuple[tuple[str, float], ...]
    reason: str


def select_window(
    audits: Iterable[Audit], *, checkpoint: str, feedback_cohort: str, step: int, window: int,
    remaining: Cost, competent: bool, policy: str = 'gain', tie_atol: float = 1e-9,
) -> Decision:
    """Select after audit costs have been charged; costs reserve one complete window.

    The gain policy needs a lower estimate of marginal unsteered improvement.
    The deficit policy is exploratory. This code neither authenticates receipts
    nor enforces hardware caps; the surrounding runner must do both.
    """
    rows = tuple(audits)
    if not checkpoint or not feedback_cohort or type(step) is not int or step < 0:
        raise ValueError('invalid current identity')
    if type(window) is not int or window <= 0 or type(competent) is not bool:
        raise ValueError('invalid window or competence flag')
    if policy not in ('gain', 'deficit') or not isfinite(tie_atol) or tie_atol < 0:
        raise ValueError('invalid selection policy')
    if len({a.action.key for a in rows}) != len(rows):
        raise ValueError('duplicate action keys')
    if any((a.checkpoint, a.feedback_cohort, a.step, a.scope) !=
           (checkpoint, feedback_cohort, step, 'feedback') for a in rows):
        raise ValueError('stale or incompatible feedback')
    if not competent:
        return Decision(None, checkpoint, feedback_cohort, step, step, (), 'competence_gate')
    ranked = []
    for row in rows:
        cost = row.window_cost_upper
        if cost is None or cost.wall_ns <= 0 or not cost.fits(remaining):
            continue
        if policy == 'gain':
            if row.gain_lower is None or row.gain_lower <= 0:
                continue
            score = row.gain_lower * (1e9 / cost.wall_ns)
        else:
            score = 1 - row.specificity / row.ideal
        if not isfinite(score):
            raise ValueError('nonfinite action score')
        ranked.append((row.action, score))
    scores = tuple(sorted((a.key, s) for a, s in ranked))
    if not ranked:
        return Decision(None, checkpoint, feedback_cohort, step, step, scores, 'no_feasible_action')
    best = max(s for _, s in ranked)
    chosen = min((a for a, s in ranked if best - s <= tie_atol), key=lambda a: a.key)
    return Decision(chosen, checkpoint, feedback_cohort, step, step + window, scores, policy)
