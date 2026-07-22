"""Architecture-neutral training-gym contracts and evaluations."""

from .comparison import compare_candidates
from .demo import run_training_gym_demo
from .evaluation import evaluate_candidate

__all__ = ["compare_candidates", "evaluate_candidate", "run_training_gym_demo"]
