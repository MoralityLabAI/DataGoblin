//! Source-only integration until built and tested with the pinned Nova crate.
//! The relying party must own the statement file and pin its exact SHA-256.
use ff::{Field, PrimeField};
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError},
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    provider::{PallasEngine, VestaEngine},
    traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait, Engine},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, env, fs, sync::Arc, time::Instant};

type E1 = PallasEngine;
type E2 = VestaEngine;
type F = <E1 as Engine>::Scalar;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, nova_snark::provider::ipa_pc::EvaluationEngine<E1>>;
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, nova_snark::provider::ipa_pc::EvaluationEngine<E2>>;
type Proof = CompressedSNARK<E1, E2, Circuit, S1, S2>;
type Row = [Vec<(usize, String)>; 3];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Statement {
    format: String,
    steps: usize,
    wires: usize,
    state_in: Vec<usize>,
    state_out: Vec<usize>,
    constraints: Vec<Row>,
    z0: Vec<String>,
    zn: Vec<String>,
}

fn field(s: &str) -> F {
    let neg = s.starts_with('-');
    let digits = if neg { &s[1..] } else { s };
    assert!(!digits.is_empty() && digits.len() <= 39 && digits.bytes().all(|c| c.is_ascii_digit()));
    assert!(digits == "0" || !digits.starts_with('0'));
    let n = F::from_str_vartime(digits).expect("decimal integer");
    if neg { -n } else { n }
}

fn vector(v: &[String]) -> Vec<F> {
    v.iter().map(|s| field(s)).collect()
}

#[derive(Clone, Debug)]
struct Circuit {
    statement: Arc<Statement>,
    advice: Vec<F>,
}

fn lc<CS: ConstraintSystem<F>>(
    mut l: LinearCombination<F>,
    terms: &[(usize, String)],
    vars: &[Option<AllocatedNum<F>>],
) -> LinearCombination<F> {
    for (i, a) in terms {
        let var = if *i == 0 { CS::one() } else { vars[*i].as_ref().unwrap().get_variable() };
        l = l + (field(a), var);
    }
    l
}

impl StepCircuit<F> for Circuit {
    fn arity(&self) -> usize {
        self.statement.state_in.len()
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self, cs: &mut CS, z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let s = &self.statement;
        let mut vars = vec![None; s.wires];
        for (&i, v) in s.state_in.iter().zip(z) {
            vars[i] = Some(v.clone());
        }
        for (i, slot) in vars.iter_mut().enumerate().skip(1) {
            if slot.is_none() {
                *slot = Some(AllocatedNum::alloc(cs.namespace(|| format!("wire_{i}")), || Ok(self.advice[i]))?);
            }
        }
        for (j, row) in s.constraints.iter().enumerate() {
            cs.enforce(
                || format!("constraint_{j}"),
                |l| lc::<CS>(l, &row[0], &vars),
                |l| lc::<CS>(l, &row[1], &vars),
                |l| lc::<CS>(l, &row[2], &vars),
            );
        }
        Ok(s.state_out.iter().map(|i| vars[*i].as_ref().unwrap().clone()).collect())
    }
}

fn read(path: &str, cap: u64) -> Vec<u8> {
    assert!(fs::metadata(path).unwrap().len() <= cap, "file limit");
    let data = fs::read(path).unwrap();
    assert!(data.len() as u64 <= cap, "file limit");
    data
}

fn load(path: &str, expected_sha: &str) -> Arc<Statement> {
    let raw = read(path, 16_000_000);
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), expected_sha, "unauthorized statement");
    let s: Statement = serde_json::from_slice(&raw).unwrap();
    assert_eq!(s.format, "zk-dg-nova-statement-v1");
    assert!((1..=32).contains(&s.steps) && (2..=50_000).contains(&s.wires));
    let arity = s.state_in.len();
    assert!((1..=67).contains(&arity));
    assert_eq!(s.state_out.len(), arity);
    assert_eq!(s.z0.len(), arity);
    assert_eq!(s.zn.len(), arity);
    assert_eq!(s.state_in.iter().copied().collect::<HashSet<_>>().len(), arity);
    assert!(s.state_in.iter().chain(&s.state_out).all(|&i| i > 0 && i < s.wires));
    assert!((1..=50_000).contains(&s.constraints.len()));
    for row in &s.constraints {
        for expr in row {
            assert!(expr.iter().all(|(i,a)| { let _ = field(a); *i < s.wires }));
        }
    }
    Arc::new(s)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 5 || args.len() == 6,
        "prove STATEMENT TRUSTED_SHA ADVICE PROOF | verify STATEMENT TRUSTED_SHA PROOF");
    let s = load(&args[2], &args[3]);
    let z0 = vector(&s.z0);
    let zn = vector(&s.zn);
    let shape = Circuit { statement: s.clone(), advice: vec![F::ZERO; s.wires] };
    let start = Instant::now();
    let pp = PublicParams::<E1, E2, Circuit>::setup(&shape, &*S1::ck_floor(), &*S2::ck_floor()).unwrap();
    eprintln!("setup_ms={} constraints={:?}", start.elapsed().as_millis(), pp.num_constraints());
    let (pk, vk) = Proof::setup(&pp).unwrap();
    if args[1] == "prove" {
        assert_eq!(args.len(), 6);
        let rows: Vec<Vec<String>> = serde_json::from_slice(&read(&args[4], 64_000_000)).unwrap();
        assert_eq!(rows.len(), s.steps);
        let circuits: Vec<Circuit> = rows.iter().map(|r| {
            assert_eq!(r.len(), s.wires);
            assert_eq!(r[0], "1");
            Circuit { statement: s.clone(), advice: vector(r) }
        }).collect();
        let start = Instant::now();
        let mut recursive = RecursiveSNARK::<E1, E2, Circuit>::new(&pp, &circuits[0], &z0).unwrap();
        for c in &circuits {
            recursive.prove_step(&pp, c).unwrap();
        }
        let folded_ms = start.elapsed().as_millis();
        assert_eq!(recursive.verify(&pp, s.steps, &z0).unwrap(), zn, "wrong recursive final state");
        let start = Instant::now();
        let proof = Proof::prove(&pp, &pk, &recursive).unwrap();
        let compress_ms = start.elapsed().as_millis();
        assert_eq!(proof.verify(&vk, s.steps, &z0).unwrap(), zn, "wrong compressed final state");
        let data = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        fs::write(&args[5], &data).unwrap();
        println!("fold_ms={folded_ms} compress_ms={compress_ms} proof_bytes={}", data.len());
    } else {
        assert_eq!(args[1], "verify");
        assert_eq!(args.len(), 5);
        let data = read(&args[4], 64_000_000);
        let (proof, used): (Proof, usize) = bincode::serde::decode_from_slice(
            &data, bincode::config::standard().with_limit::<64_000_000>()).unwrap();
        assert_eq!(used, data.len(), "trailing proof bytes");
        let start = Instant::now();
        assert_eq!(proof.verify(&vk, s.steps, &z0).unwrap(), zn, "wrong compressed final state");
        println!("accepted verify_ms={}", start.elapsed().as_millis());
    }
}
