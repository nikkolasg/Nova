use crate::traits::circuit::TrivialTestCircuit;
use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct MapReduceArity(usize, usize);
impl MapReduceArity {
  pub fn new(map_in: usize, map_out: usize) -> Self {
    Self(map_in, map_out)
  }
  pub fn map_input(&self) -> usize {
    self.0
  }
  pub fn map_output(&self) -> usize {
    self.1
  }
  // Reduce steps take 2 inputs and produce one
  // These inputs come either from the map step or the reduce step
  // Therefore, input of reduce is twice the output of map step
  // and output of reduce is only once output of map step
  pub fn reduce_input(&self) -> usize {
    self.1 * 2
  }
  pub fn reduce_output(&self) -> usize {
    self.1
  }
  // the inputs of one step is the max of the reduce or map step because currently
  // we use a single circuit to operate both. With Supernova, a different circuit
  // can happen for each step.
  // This is necessary to allocate the input no matter a which steps we are
  pub fn total_input(&self) -> usize {
    std::cmp::max(self.reduce_input(), self.map_input())
  }
  pub fn total_output(&self) -> usize {
    self.1
  }
}
pub trait MapReduceCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the number of (input / output) the map function takes.
  /// (this method is called only at circuit synthesis time)
  /// `synthesize_map` and `output_map` methods's argument are expected to
  /// match these sizes
  /// The reduce step has input and output arity equal to the map output arity.
  fn arity(&self) -> MapReduceArity;

  fn synthesize_map<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;

  fn synthesize_reduce<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z_left: &[AllocatedNum<F>],
    z_right: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;

  fn output_map(&self, z: &[F]) -> Vec<F>;
  fn output_reduce(&self, z_left: &[F], z_right: &[F]) -> Vec<F>;
}

impl<F> MapReduceCircuit<F> for TrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> MapReduceArity {
    MapReduceArity(1, 1)
  }

  fn synthesize_map<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    Ok(z.to_vec())
  }

  fn synthesize_reduce<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z_left: &[AllocatedNum<F>],
    z_right: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    Ok(z_left.to_vec())
  }

  fn output_map(&self, z: &[F]) -> Vec<F> {
    z.to_vec()
  }
  fn output_reduce(&self, z_left: &[F], z_right: &[F]) -> Vec<F> {
    z_left.to_vec()
  }
}
