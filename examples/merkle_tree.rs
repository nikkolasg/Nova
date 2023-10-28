use bellperson::SynthesisError;
use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem};
use eyre::Result as EResult;
use ff::Field;
use ff::PrimeField;
use flate2::{write::ZlibEncoder, Compression};
use generic_array::typenum::U2;
use neptune::{circuit2::poseidon_hash_allocated, poseidon::PoseidonConstants};
use nova_snark::errors::NovaError;
use nova_snark::mapreduce::prover::PublicParams;
use nova_snark::traits::Group;
use nova_snark::{
  mapreduce::prover::TreeNode,
  mapreduce::{
    self,
    circuit::{MapReduceArity, MapReduceCircuit},
  },
  traits::circuit::TrivialTestCircuit,
};
use rayon::prelude::*;
use serde::Serializer;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

// Circuit that recursively and in parallel proves the building of a Merkle tree
// using a generic hash function.
#[derive(Clone)]
struct BinaryMTCircuit<F: PrimeField, H: Hasher<F>> {
  h: H,
  _f: PhantomData<F>,
}
impl<F, H> BinaryMTCircuit<F, H>
where
  F: PrimeField,
  H: Hasher<F>,
{
  pub fn new(h: H) -> Self {
    Self { h, _f: PhantomData }
  }
}
pub trait Hasher<F: PrimeField>: Clone + Send + Sync {
  fn hash(&self, left: F, right: F) -> EResult<F>;
  fn hash_circuit<CS: ConstraintSystem<F>>(
    &self,
    cs: CS,
    left: AllocatedNum<F>,
    right: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError>;
}

impl<F, H> MapReduceCircuit<F> for BinaryMTCircuit<F, H>
where
  F: PrimeField,
  H: Hasher<F>,
{
  fn arity(&self) -> MapReduceArity {
    // implementation only folds two instances together so we are stuck with arity 2
    MapReduceArity::new(1, 1)
  }
  fn synthesize_map<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    Ok(vec![z[0].clone()])
  }

  fn synthesize_reduce<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z_left: &[AllocatedNum<F>],
    z_right: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    self
      .h
      .hash_circuit(
        cs.namespace(|| "merkle binary tree"),
        z_left[0].clone(),
        z_right[0].clone(),
      )
      .map(|e| vec![e])
  }
  fn output_map(&self, z: &[F]) -> Vec<F> {
    z.to_vec()
  }
  fn output_reduce(&self, z_left: &[F], z_right: &[F]) -> Vec<F> {
    let output = self
      .h
      .hash(z_left[0].clone(), z_right[0].clone())
      .expect("hashing outside circuit failed");
    vec![output]
  }
}

// Poseidon implementation of a Hasher
#[derive(Clone)]
struct PoseidonHasher<F: PrimeField> {
  c: PoseidonConstants<F, U2>,
}
impl<F> PoseidonHasher<F>
where
  F: PrimeField,
{
  pub fn new() -> Self {
    let c: PoseidonConstants<F, U2> = PoseidonConstants::new();
    Self { c }
  }
}

impl<F> Hasher<F> for PoseidonHasher<F>
where
  F: PrimeField,
{
  fn hash(&self, left: F, right: F) -> EResult<F> {
    let mut h = neptune::Poseidon::<F, U2>::new(&self.c);
    h.input(left)?;
    h.input(right)?;
    Ok(h.hash())
  }
  fn hash_circuit<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    left: AllocatedNum<F>,
    right: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    poseidon_hash_allocated(
      cs.namespace(|| "H(left||right)"),
      vec![left, right],
      &self.c,
    )
  }
}

// build the tree outside the circuit, "normally"
fn build_binary_tree_root<F: PrimeField, H: Hasher<F>>(hasher: &H, leaves: Vec<F>) -> EResult<F> {
  let mut nodes = leaves;
  while nodes.len() > 1 {
    nodes = nodes
      .par_chunks(2)
      .map(|pair| match pair {
        [left, right] => hasher.hash(left.clone(), right.clone()),
        [single] => Ok(single.clone()),
        _ => panic!("that should not happen"),
      })
      .collect::<EResult<Vec<_>>>()?;
  }
  Ok(nodes[0].clone())
}

// Build the folding tree and returns the treenode representing the root
// Second circuit hardcoded to be trivial
fn build_root_instance<'a, G1, G2, C1>(
  pp: &'a PublicParams<G1, G2, C1, TrivialTestCircuit<G2::Scalar>>,
  c1: C1,
  leaves: Vec<G1::Scalar>,
) -> Result<TreeNode<'a, G1, G2, C1, TrivialTestCircuit<G2::Scalar>>, NovaError>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
{
  let z0s_secondary = (0..leaves.len())
    .map(|_| vec![G2::Scalar::one()])
    .collect::<Vec<_>>();
  let z0s_primary = leaves.into_iter().map(|v| vec![v]).collect::<Vec<_>>();
  mapreduce::prover::vec_map_reduce(
    &pp,
    z0s_primary,
    z0s_secondary,
    c1,
    TrivialTestCircuit::default(),
  )
}

fn gen_random_leaves<F: PrimeField>(n: usize) -> Vec<F> {
  (0..n).map(|_| F::random(&mut rand::thread_rng())).collect()
}

fn duration_to_ms<S>(x: &Duration, s: S) -> Result<S::Ok, S::Error>
where
  S: Serializer,
{
  s.serialize_u64(x.as_millis() as u64)
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub enum HashFunction {
  #[default]
  Poseidon,
}
#[derive(Clone, Debug, Default, serde::Serialize)]
struct BenchParams {
  pub hash: HashFunction,
  pub nleaves: usize,
}

#[derive(Clone, Debug, serde::Serialize, Default)]
struct BenchResult {
  #[serde(flatten)]
  pub params: BenchParams,
  nconstraints: usize,
  nconstraints2: usize, // secondary circuit
  #[serde(serialize_with = "duration_to_ms")]
  prover_setup: Duration,
  #[serde(serialize_with = "duration_to_ms")]
  prover_build_tree: Duration,
  #[serde(serialize_with = "duration_to_ms")]
  prover_recursion: Duration,
  witness_size: usize,
  #[serde(serialize_with = "duration_to_ms")]
  verification_time: Duration,
}

fn run_experiment(p: BenchParams) -> EResult<BenchResult> {
  let mut res = BenchResult {
    params: p.clone(),
    ..BenchResult::default()
  };
  let n = p.nleaves;
  let leaves = gen_random_leaves::<F1>(n);
  let hasher = match p.hash {
    HashFunction::Poseidon => PoseidonHasher::<F1>::new(),
  };

  let start = Instant::now();
  let root = build_binary_tree_root(&hasher, leaves.clone())?;
  res.prover_build_tree += start.elapsed();
  let circuit = BinaryMTCircuit::new(hasher.clone());
  let start = Instant::now();
  let pp = PublicParams::<G1, G2, _, _>::setup(circuit.clone(), TrivialTestCircuit::default())?;
  res.nconstraints = pp.num_constraints().0;
  res.nconstraints2 = pp.num_constraints().1;
  res.prover_setup += start.elapsed();
  let start = Instant::now();
  let final_instance = build_root_instance::<G1, G2, BinaryMTCircuit<F1, PoseidonHasher<F1>>>(
    &pp,
    circuit.clone(),
    leaves,
  )?;
  res.prover_recursion += start.elapsed();
  let start = Instant::now();
  let (root_circuit, _) = final_instance.verify()?;
  assert!(root_circuit[0] == root);
  res.verification_time += start.elapsed();

  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serialize_into(&mut encoder, &final_instance.data).unwrap();
  let buffer = encoder.finish().unwrap();
  res.witness_size += buffer.len();
  Ok(res)
}

type G1 = pasta_curves::pallas::Point;
type F1 = pasta_curves::pallas::Scalar;
type G2 = pasta_curves::vesta::Point;

fn main() -> EResult<()> {
  println!("========================================");
  println!("Parallel Binary Tree Building MapReduce");
  println!("========================================");
  let fname = "bench.csv";
  println!("[+] Writing results to fresh {} file", &fname);
  let mut wtr = csv::Writer::from_writer(std::fs::File::create(&fname)?);
  let leaves = vec![8, 16];
  for n in leaves {
    let params = BenchParams {
      hash: HashFunction::Poseidon,
      nleaves: n,
    };
    println!("[+] Running experiment for {:?}", params);
    let res = run_experiment(params)?;
    wtr.serialize(res)?;
  }
  Ok(())
}
