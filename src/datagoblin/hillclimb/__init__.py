"""Hill-climb controller surface for the DataGoblin GDK app.

The app's local model proposes moves; this package owns everything that decides them: the lane
registry and its frozen gates, the allowlisted recipes, the hash-chained ledger, curation, the
Agonist worker that asks bigger local models for eval worlds, and the decision bench used to train
an adapter on the driver model itself.
"""

from .lanes import TIERS, HillclimbError, Registry

__all__ = ["TIERS", "HillclimbError", "Registry"]
