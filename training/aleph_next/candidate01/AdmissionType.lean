import Lean

open Lean Elab Command

namespace AlephAdmission

/-- A structural type serialization. Bound variables use de Bruijn indices.
    Binder names and annotations do not contribute; term constructors and universes do. -/
partial def typeCode : Expr → String
  | .bvar i => "b" ++ toString i
  | .sort u => "s" ++ reprStr u
  | .const n us => "c" ++ reprStr n ++ reprStr us
  | .app f x => "(a " ++ typeCode f ++ " " ++ typeCode x ++ ")"
  | .lam _ x y _ => "(l " ++ typeCode x ++ " " ++ typeCode y ++ ")"
  | .forallE _ x y _ => "(p " ++ typeCode x ++ " " ++ typeCode y ++ ")"
  | .letE _ t v b _ => "(e " ++ typeCode t ++ " " ++ typeCode v ++ " " ++ typeCode b ++ ")"
  | .lit n => "n" ++ reprStr n
  | .mdata _ e => typeCode e
  | .proj n i x => "(j " ++ reprStr n ++ " " ++ toString i ++ " " ++ typeCode x ++ ")"
  | .fvar id => "f" ++ reprStr id
  | .mvar id => "m" ++ reprStr id

elab "#signature " name:ident bs:bracketedBinder* " : " body:term : command => do
  let val ← if bs.isEmpty then pure body else `(∀ $bs*, $body)
  let text ← liftTermElabM do
    let ty ← Term.elabType val
    Term.synthesizeSyntheticMVarsNoPostponing
    let ty ← instantiateMVars ty
    if ty.hasMVar || ty.hasFVar then throwError "expected a closed type expression"
    pure ((typeCode ty).replace "\n" " ")
  liftIO <| IO.println ("TYPED " ++ toString name.getId ++ " " ++ text)

end AlephAdmission
