import Init

theorem dgOxiIdentity (p : Prop) (h : p) : p := h
theorem dgOxiPairSwap (p q : Prop) (h : p ∧ q) : q ∧ p := ⟨h.2, h.1⟩
theorem dgOxiEitherLeft (p q : Prop) (h : p) : p ∨ q := Or.inl h
theorem dgOxiEitherRight (p q : Prop) (h : q) : p ∨ q := Or.inr h
theorem dgOxiTransport (a b : Nat) (h : a = b) : b = a := h.symm
theorem dgOxiCompose (p q r : Prop) (f : p → q) (g : q → r) (h : p) : r := g (f h)
theorem dgOxiUnit : True := True.intro
theorem dgOxiRefl (xs : List Nat) : xs = xs := rfl

#print axioms dgOxiIdentity
#print axioms dgOxiPairSwap
#print axioms dgOxiEitherLeft
#print axioms dgOxiEitherRight
#print axioms dgOxiTransport
#print axioms dgOxiCompose
#print axioms dgOxiUnit
#print axioms dgOxiRefl
