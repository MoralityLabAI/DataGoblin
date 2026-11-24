"""Two-pass dense Aleph reference models for the Inky Mini architecture note.

Pass a single causal segment per row, or use packing.reference_logits for a
multi-document pack. Construction is identical for both arms.
"""

from gym import Aleph
from torch import lerp
from torch.nn import functional as F


class InkyPair(Aleph):
    def forward(self, tokens, arm="direct", r=2, s=1):
        if (r, s) != (2, 1):
            raise ValueError("the matched pair uses two core passes and one answer pass")
        if arm == "direct":
            return super().forward(tokens, arm, r, s)
        if arm != "prelude_reread":
            raise ValueError(f"unknown Inky pair arm: {arm}")

        prelude = self.blocks[0](self.embed(tokens))
        x = prelude
        for visit in range(2):
            for physical in range(1, 11):
                if visit == 1 and physical == 6:
                    x = lerp(x, prelude, 0.25)
                x = self.blocks[physical](x)
        for physical in (11, 12, 13):
            x = self.blocks[physical](x)
        return F.linear(self.norm(x), self.embed.weight)
