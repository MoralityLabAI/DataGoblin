# What the source and clock interventions measure

Write a shared two-block pass as $C=V\circ U$ and the stored prelude as $p$. If $x$ is the input to visit two, the immediate inputs to $V$ under fixed source and local rereads differ by

$$
[(1-\beta)U(x)+\beta p]-[(1-\beta)U(x)+\beta x]=\beta(p-x).
$$

This equality holds at fixed weights and fixed incoming state. If both anchors coincide, the interventions agree exactly. For a differentiable $V$, their endpoint difference is the line integral of $DV$ along the segment between the two inputs, applied to $\beta(p-x)$. Departures between independently fitted checkpoints also include optimization. The implementation therefore exchanges the anchor at fixed learned weights on the same held examples in addition to comparing the trained arms.

Let the whole-pass proposal be $C(x)$ and write $D(x)=C(x)-x$. A scalar-clock write is $T(x)=x+a(x)D(x)$. With a fixed coefficient, $DT=I+a(DC-I)$. With a state-dependent coefficient,

$$
DT=I+a(DC-I)+D\,(\nabla a)^T.
$$

In the dense proxy the gate is scalar per token. Each token's Jacobian row block gets its own outer-product term. Its derivative may depend on the causally preceding tokens through the pass. For $a=\sigma(b+k m)$ with $m=\log(1+\lVert D\rVert/\lVert x\rVert)$, away from zero norms the contribution per row block has operator norm at most $\lVert D\rVert |k|\lVert\nabla m\rVert/4$. This explains why learning to damp strong corrections can change stability in either direction. The visit-only learned clock has the same two scalar parameters and its coefficient depends only on the pass number.

For a fixed linear pass $J$, set $M=J-I$. In three visits the common final factor is $J$. The first two RLT factors have the same sum of coefficients as constant damping, while their product is $1/6$ versus $25/144$. The exact endpoint difference on input $x$ is

$$
J\left[(I+M/2)(I+M/3)-(I+5M/12)^2\right]x=-JM^2x/144.
$$

For three fixed linear visit maps $J_1,J_2,J_3$, put $M_i=J_i-I$. Expanding the ordered product gives $J_3[(M_2-M_1)/12-M_2M_1/144]$. The first-order term can survive when visits differ. For a nonlinear shared core, different schedules visit different states, so substituting three estimated Jacobians into this expression does not give an exact nonlinear endpoint difference.

This cancellation makes a large latent displacement insufficient by itself to show a clock advantage. For example, a translational affine residual has constant displacement and zero curvature, and the schedules agree exactly. The relevant empirical comparison is the RLT-minus-damping verified advantage at high correction, paired with its clean counterpart. The assay also measures conditional behavior after switching clocks on the same fixed weights, local directional curvature, normalized proposal and realized displacement, successive-update alignment and external task success per visit.

The active perturbation is applied after the first completed visit. Let $u_t$ be a direction of unit feature RMS at token $t$, and take $z'_t=z_t+s\operatorname{RMS}(z_t)u_t$ for $s\in\{0,.5,1.25\}$. Summing squared feature norms across tokens gives $\lVert z'-z\rVert_F=s\lVert z\rVert_F$ for nonzero directions. The same equality holds for the targeted and randomized directions independently at a given state. A source-trained checkpoint therefore faces comparably large perturbations with different alignment. The targeted direction is constructed from the printed claims alone. [Primary registration](ARCH_PROTOCOL.md) · [Basis-sweep registration](ARCH_SPECTRAL_PROTOCOL.md)
