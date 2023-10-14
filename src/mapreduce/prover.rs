#![allow(unused_imports)]
#![allow(unused)]
//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! Each recursive tree node has both an aggregated and new instance
//! of both the primary and secondary circuit. As you merge the nodes
//! the proofs verify that three folding are correct and merge the
//! running instances and new instances in a pair of nodes to a single
//! running instance.
//! We check that hash(index start, index end, z_start, z_end) has been
//! committed properly for each node.
//! The circuit also checks that when F is executed on the left nodes
//! z_end that the output is z_start of the right node

use crate::{
  bellperson::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  circuit::NovaAugmentedCircuitParams,
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    utils::{alloc_num_equals, alloc_scalar_as_base, conditionally_select_vec, le_bits_to_num},
  },
  mapreduce::{
    circuit::{MapReduceArity, MapReduceCircuit},
    folding::{NovaAugmentedParallelCircuit, NovaAugmentedParallelCircuitInputs},
  },
  nifs::NIFS,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, R1CS},
  traits::{
    circuit::TrivialTestCircuit,
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    snark::RelaxedR1CSSNARKTrait,
    AbsorbInROTrait, Group, ROConstants, ROConstantsCircuit, ROConstantsTrait, ROTrait,
  },
  Commitment,
};
use ark_std::{end_timer, start_timer};
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    test::TestConstraintSystem,
    Assignment,
  },
  Circuit, ConstraintSystem, Index, SynthesisError,
};
use core::marker::PhantomData;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// TODO - This is replicated from lib but we should actually instead have another file for it and use both here and there

type CommitmentKey<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey;

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
  C2: MapReduceCircuit<G2::Scalar>,
{
  F_arity_primary: MapReduceArity,
  F_arity_secondary: MapReduceArity,
  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<G2>,
  ck_primary: CommitmentKey<G1>,
  r1cs_shape_primary: R1CSShape<G1>,
  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<G1>,
  ck_secondary: CommitmentKey<G2>,
  r1cs_shape_secondary: R1CSShape<G2>,
  augmented_circuit_params_primary: NovaAugmentedCircuitParams,
  augmented_circuit_params_secondary: NovaAugmentedCircuitParams,
  _p_c1: PhantomData<C1>,
  _p_c2: PhantomData<C2>,
}

impl<G1, G2, C1, C2> PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
  C2: MapReduceCircuit<G2::Scalar>,
{
  /// Create a new `PublicParams`
  pub fn setup(c_primary: C1, c_secondary: C2) -> Result<Self, NovaError> {
    let augmented_circuit_params_primary =
      NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let augmented_circuit_params_secondary =
      NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::new();
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::new();

    let F_arity_primary = c_primary.arity();
    let F_arity_secondary = c_secondary.arity();

    // ro_consts_circuit_primary are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<G2> = ROConstantsCircuit::<G2>::new();
    let ro_consts_circuit_secondary: ROConstantsCircuit<G1> = ROConstantsCircuit::<G1>::new();

    // Initialize ck for the primary
    let circuit_primary: NovaAugmentedParallelCircuit<G2, C1> = NovaAugmentedParallelCircuit::new(
      augmented_circuit_params_primary.clone(),
      None,
      c_primary,
      ro_consts_circuit_primary.clone(),
    );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    println!("BEFORE FIRST SNYTHESIS");
    {
      let mut tcs = TestConstraintSystem::new();
      let _ = circuit_primary.clone().synthesize(&mut tcs)?;
      println!("IS satisfied? {}", tcs.is_satisfied());
      println!("Which is unsatisfied? {:?}", tcs.which_is_unsatisfied());
    }
    let _ = circuit_primary
      .synthesize(&mut cs)
      .map_err(|e| NovaError::SynthesisError(e))?;
    let (r1cs_shape_primary, ck_primary) = cs.r1cs_shape();
    println!("PASSED FIRST SNYTHESIS");
    // Initialize ck for the secondary
    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      augmented_circuit_params_secondary.clone(),
      None,
      c_secondary,
      ro_consts_circuit_secondary.clone(),
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary
      .synthesize(&mut cs)
      .map_err(|e| NovaError::SynthesisError(e))?;
    let (r1cs_shape_secondary, ck_secondary) = cs.r1cs_shape();

    Ok(Self {
      F_arity_primary,
      F_arity_secondary,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary,
      r1cs_shape_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary,
      r1cs_shape_secondary,
      augmented_circuit_params_primary,
      augmented_circuit_params_secondary,
      _p_c1: Default::default(),
      _p_c2: Default::default(),
    })
  }

  /// Returns the number of constraints in the primary and secondary circuits
  pub fn num_constraints(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_cons,
      self.r1cs_shape_secondary.num_cons,
    )
  }

  /// Returns the number of variables in the primary and secondary circuits
  pub fn num_variables(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_vars,
      self.r1cs_shape_secondary.num_vars,
    )
  }
}

// This ends the 1 to 1 copied code

#[derive(Clone)]
pub struct TreeNode<'a, G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
  C2: MapReduceCircuit<G2::Scalar>,
{
  pub data: NodeData<G1, G2>,
  pp: &'a PublicParams<G1, G2, C1, C2>,
  c_primary: C1,
  c_secondary: C2,
}

/// A type that holds one node the tree based nova proof. This will have both running instances and fresh instances
/// of the primary and secondary circuit.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NodeData<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  // The running instance of the primary
  W_primary: RelaxedR1CSWitness<G1>,
  U_primary: RelaxedR1CSInstance<G1>,
  // The new instance of the primary
  w_primary: RelaxedR1CSWitness<G1>,
  u_primary: RelaxedR1CSInstance<G1>,
  // The running instance of the secondary
  W_secondary: RelaxedR1CSWitness<G2>,
  U_secondary: RelaxedR1CSInstance<G2>,
  // The running instance of the secondary
  w_secondary: RelaxedR1CSWitness<G2>,
  u_secondary: RelaxedR1CSInstance<G2>,
  i_start: u64,
  i_end: u64,
  z_start_primary: Vec<G1::Scalar>,
  z_end_primary: Vec<G1::Scalar>,
  z_start_secondary: Vec<G2::Scalar>,
  z_end_secondary: Vec<G2::Scalar>,
}

impl<'a, G1, G2, C1, C2> TreeNode<'a, G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
  C2: MapReduceCircuit<G2::Scalar>,
{
  /// Creates a leaf tree node which proves one computation and runs a base case F' proof. The running instances
  /// are set to defaults and the new proofs are set ot this base case proof.
  pub fn map_step(
    pp: &'a PublicParams<G1, G2, C1, C2>,
    c_primary: C1,
    c_secondary: C2,
    i: u64,
    z_start_primary: Vec<G1::Scalar>,
    z_start_secondary: Vec<G2::Scalar>,
  ) -> Result<Self, NovaError> {
    // base case for the primary
    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: NovaAugmentedParallelCircuitInputs<G2> =
      NovaAugmentedParallelCircuitInputs::new(
        pp.r1cs_shape_secondary.get_digest(),
        G1::Scalar::from(i.try_into().unwrap()),
        G1::Scalar::from((i).try_into().unwrap()),
        // we do this to satisfy the constraint in the augmented circuit
        // that left_end == right_start which is in the base case as well
        G1::Scalar::from((i + 1).try_into().unwrap()),
        G1::Scalar::from((i + 1).try_into().unwrap()),
        z_start_primary.clone(),
        z_start_primary.clone(),
        z_start_primary.clone(),
        z_start_primary.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
      );

    let circuit_primary: NovaAugmentedParallelCircuit<G2, C1> = NovaAugmentedParallelCircuit::new(
      pp.augmented_circuit_params_primary.clone(),
      Some(inputs_primary),
      c_primary.clone(),
      pp.ro_consts_circuit_primary.clone(),
    );
    let _ = circuit_primary.synthesize(&mut cs_primary)?;
    let (u_primary, w_primary) =
      cs_primary.r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)?;

    // base case for the secondary
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();

    let inputs_secondary: NovaAugmentedParallelCircuitInputs<G1> =
      NovaAugmentedParallelCircuitInputs::new(
        pp.r1cs_shape_primary.get_digest(),
        G2::Scalar::from(i),
        G2::Scalar::from(i),
        G2::Scalar::from(i + 1),
        G2::Scalar::from(i + 1),
        z_start_secondary.clone(),
        z_start_secondary.clone(),
        z_start_secondary.clone(),
        z_start_secondary.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
      );
    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      pp.augmented_circuit_params_secondary.clone(),
      Some(inputs_secondary),
      c_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);
    let (u_secondary, w_secondary) =
      cs_secondary.r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)?;

    // IVC proof for the primary circuit
    let w_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &w_primary);
    let u_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck_primary, &pp.r1cs_shape_primary, &u_primary);
    let W_primary = w_primary.clone();
    let U_primary = u_primary.clone();

    // IVC proof of the secondary circuit
    let w_secondary =
      RelaxedR1CSWitness::<G2>::from_r1cs_witness(&pp.r1cs_shape_secondary, &w_secondary);
    let u_secondary = RelaxedR1CSInstance::<G2>::from_r1cs_instance(
      &pp.ck_secondary,
      &pp.r1cs_shape_secondary,
      &u_secondary,
    );
    let W_secondary = w_secondary.clone();
    let U_secondary = u_secondary.clone();

    if z_start_primary.len() != pp.F_arity_primary.total_input()
      || z_start_secondary.len() != pp.F_arity_secondary.total_input()
    {
      return Err(NovaError::InvalidStepOutputLength);
    }

    let i_start = i;
    let i_end = i + 1;
    // base case is map step
    let z_end_primary = c_primary.output_map(&z_start_primary);
    let z_end_secondary = c_secondary.output_map(&z_start_secondary);

    Ok(Self {
      pp: pp,
      c_primary: c_primary,
      c_secondary: c_secondary,
      data: NodeData {
        W_primary,
        U_primary,
        w_primary,
        u_primary,
        W_secondary,
        U_secondary,
        w_secondary,
        u_secondary,
        i_start,
        i_end,
        z_start_primary,
        z_end_primary,
        z_start_secondary,
        z_end_secondary,
      },
    })
  }

  /// Merges another node into this node. The node this is called on is treated as the left node and the node which is
  /// consumed is treated as the right node.
  pub fn reduce(self, nright: TreeNode<'a, G1, G2, C1, C2>) -> Result<Self, NovaError> {
    let left = self.data;
    let right = nright.data;
    // NOTE: a major difference with the original PSE code is that we do not let a "space" between
    // two leaves. All leaves are assigned consecutive indices.
    // Ex: base leaf [0->1] [1->2] , when    // Ex: base leaf [0->1] [1->2] , when we merge, final indices are [0->2]]
    if left.i_end != right.i_start {
      return Err(NovaError::InvalidNodeMerge);
    }

    // First we fold the secondary instances of both the left and right children in the secondary curve
    let (nifs_left_secondary, (left_U_secondary, left_W_secondary)) = NIFS::prove(
      &self.pp.ck_secondary,
      &self.pp.ro_consts_secondary,
      &self.pp.r1cs_shape_secondary,
      &left.U_secondary,
      &left.W_secondary,
      &left.u_secondary,
      &left.w_secondary,
      false,
    )?;
    let (nifs_right_secondary, (right_U_secondary, right_W_secondary)) = NIFS::prove(
      &self.pp.ck_secondary,
      &self.pp.ro_consts_secondary,
      &self.pp.r1cs_shape_secondary,
      &right.U_secondary,
      &right.W_secondary,
      &right.u_secondary,
      &right.w_secondary,
      false,
    )?;
    // here we're folding two relaxed R1CS instances instead of the vanilla folding step of Nova RR1CS x R1CS
    let (nifs_secondary, (U_secondary, W_secondary)) = NIFS::prove(
      &self.pp.ck_secondary,
      &self.pp.ro_consts_secondary,
      &self.pp.r1cs_shape_secondary,
      &left_U_secondary,
      &left_W_secondary,
      &right_U_secondary,
      &right_W_secondary,
      true,
    )?;

    // Next we construct a proof of this folding and of the invocation of F

    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();

    let inputs_primary: NovaAugmentedParallelCircuitInputs<G2> =
      NovaAugmentedParallelCircuitInputs::new(
        self.pp.r1cs_shape_secondary.get_digest(),
        G1::Scalar::from(left.i_start as u64),
        G1::Scalar::from(left.i_end as u64),
        G1::Scalar::from(right.i_start as u64),
        G1::Scalar::from(right.i_end as u64),
        left.z_start_primary.clone(),
        left.z_end_primary.clone(),
        right.z_start_primary,
        right.z_end_primary.clone(),
        Some(left.U_secondary),
        Some(left.u_secondary),
        Some(right.U_secondary),
        Some(right.u_secondary),
        Some(Commitment::<G2>::decompress(&nifs_left_secondary.comm_T)?),
        Some(Commitment::<G2>::decompress(&nifs_right_secondary.comm_T)?),
        Some(Commitment::<G2>::decompress(&nifs_secondary.comm_T)?),
      );

    let circuit_primary: NovaAugmentedParallelCircuit<G2, C1> = NovaAugmentedParallelCircuit::new(
      self.pp.augmented_circuit_params_primary.clone(),
      Some(inputs_primary),
      self.c_primary.clone(),
      self.pp.ro_consts_circuit_primary.clone(),
    );
    let _ = circuit_primary.synthesize(&mut cs_primary);

    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&self.pp.r1cs_shape_primary, &self.pp.ck_primary)
      .map_err(|_e| NovaError::UnSat)?;

    let u_primary = RelaxedR1CSInstance::from_r1cs_instance(
      &self.pp.ck_primary,
      &self.pp.r1cs_shape_primary,
      &u_primary,
    );
    let w_primary = RelaxedR1CSWitness::from_r1cs_witness(&self.pp.r1cs_shape_primary, &w_primary);

    // Now we fold the instances of the primary proof
    let (nifs_left_primary, (left_U_primary, left_W_primary)) = NIFS::prove(
      &self.pp.ck_primary,
      &self.pp.ro_consts_primary,
      &self.pp.r1cs_shape_primary,
      &left.U_primary,
      &left.W_primary,
      &left.u_primary,
      &left.w_primary,
      false,
    )?;
    let (nifs_right_primary, (right_U_primary, right_W_primary)) = NIFS::prove(
      &self.pp.ck_primary,
      &self.pp.ro_consts_primary,
      &self.pp.r1cs_shape_primary,
      &right.U_primary,
      &right.W_primary,
      &right.u_primary,
      &right.w_primary,
      false,
    )?;
    let (nifs_primary, (U_primary, W_primary)) = NIFS::prove(
      &self.pp.ck_primary,
      &self.pp.ro_consts_primary,
      &self.pp.r1cs_shape_primary,
      &left_U_primary,
      &left_W_primary,
      &right_U_primary,
      &right_W_primary,
      true,
    )?;

    // Next we construct a proof of this folding in the secondary curve
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();

    let inputs_secondary: NovaAugmentedParallelCircuitInputs<G1> =
      NovaAugmentedParallelCircuitInputs::<G1>::new(
        self.pp.r1cs_shape_primary.get_digest(),
        G2::Scalar::from(left.i_start as u64),
        G2::Scalar::from(left.i_end as u64),
        G2::Scalar::from(right.i_start as u64),
        G2::Scalar::from(right.i_end as u64),
        left.z_start_secondary.clone(),
        left.z_end_secondary.clone(),
        right.z_start_secondary,
        right.z_end_secondary.clone(),
        Some(left.U_primary),
        Some(left.u_primary),
        Some(right.U_primary),
        Some(right.u_primary),
        Some(Commitment::<G1>::decompress(&nifs_left_primary.comm_T)?),
        Some(Commitment::<G1>::decompress(&nifs_right_primary.comm_T)?),
        Some(Commitment::<G1>::decompress(&nifs_primary.comm_T)?),
      );

    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      self.pp.augmented_circuit_params_secondary.clone(),
      Some(inputs_secondary),
      self.c_secondary.clone(),
      self.pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);

    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&self.pp.r1cs_shape_secondary, &self.pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // Give these a trivial error vector
    let u_secondary = RelaxedR1CSInstance::from_r1cs_instance(
      &self.pp.ck_secondary,
      &self.pp.r1cs_shape_secondary,
      &u_secondary,
    );
    let w_secondary =
      RelaxedR1CSWitness::from_r1cs_witness(&self.pp.r1cs_shape_secondary, &w_secondary);

    // here we take minimum value of the left instance and the max of the right instance
    // we know we have proven for this range.
    let i_start = left.i_start.clone();
    let i_end = right.i_end.clone();
    // z_start is technically useless after the map step, because reduce always take
    // the output and do not hash the first values. Nevertheless since we are in a single circuit
    // we have to give it as input always, as the "map" function is always computed.
    // TODO: for Supernova, it probably can be removed.
    let mut z_start_primary = left.z_start_primary;
    // Compute the output ourselves Reduce(left_node.output,right_node.output)
    let z_end_primary = self
      .c_primary
      .output_reduce(&left.z_end_primary, &right.z_end_primary);
    assert!(
      z_end_primary.len() != self.c_primary.arity().reduce_output(),
      "Inconsistent reduce output lengths"
    );
    let z_start_secondary = left.z_start_secondary;
    let z_end_secondary = self
      .c_secondary
      .output_reduce(&left.z_end_secondary, &right.z_end_secondary);
    assert!(
      z_end_secondary.len() != self.c_secondary.arity().reduce_output(),
      "Inconsistent reduce output lengths"
    );

    Ok(TreeNode {
      pp: self.pp,
      c_primary: self.c_primary,
      c_secondary: self.c_secondary,
      data: NodeData {
        // Primary running instance
        W_primary,
        U_primary,
        // Primary new instance
        w_primary,
        u_primary,
        // The running instance of the secondary
        W_secondary,
        U_secondary,
        // The running instance of the secondary
        w_secondary,
        u_secondary,
        // The range data
        i_start,
        i_end,
        z_start_primary,
        z_end_primary,
        z_start_secondary,
        z_end_secondary,
      },
    })
  }
}

/// Structure for parallelization
#[derive(Clone)]
pub struct ParallelSNARK<'a, G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
  C2: MapReduceCircuit<G2::Scalar>,
{
  pp: &'a PublicParams<G1, G2, C1, C2>,
  nodes: Vec<TreeNode<'a, G1, G2, C1, C2>>,
}

/// Implementation for parallelization SNARK
impl<'a, G1, G2, C1, C2> ParallelSNARK<'a, G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: MapReduceCircuit<G1::Scalar>,
  C2: MapReduceCircuit<G2::Scalar>,
{
  /// Create a new instance of parallel SNARK
  pub fn prove(
    pp: &'a PublicParams<G1, G2, C1, C2>,
    // Each z0 for each leaf, independent
    z0s_primary: Vec<Vec<G1::Scalar>>,
    z0_secondary: Vec<Vec<G2::Scalar>>,
    c_primary: C1,
    c_secondary: C2,
  ) -> Result<Self, NovaError> {
    let leaves = z0s_primary
      .into_par_iter()
      .zip(z0_secondary)
      .enumerate()
      .map(|(i, (z0p, z0s))| {
        TreeNode::map_step(
          &pp,
          c_primary.clone(),
          c_secondary.clone(),
          i as u64,
          z0p,
          z0s,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    // Calculate the max height of the tree
    // ⌈log2(n)⌉ + 1
    let max_height = ((leaves.len() as f64).log2().ceil() + 1f64) as usize;

    let mut nodes = leaves;
    // Build up the tree with max given height
    for level in 0..max_height {
      // Exist if we on the root of the tree
      if nodes.len() == 1 {
        break;
      }
      // New nodes list will reduce a half each round
      nodes = nodes
        .par_chunks(2)
        .map(|item| match item {
          // There are 2 nodes in the chunk
          [vl, vr] => vl.clone()
          // TODO remove these clones maybe with iter tools...
            .reduce(vr.clone()),
          // Just 1 node left, we carry it to the next level
          [vl] => Ok(vl.clone()),
          _ => panic!("Invalid chunk size - tree of size not power of two not supported yet"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    }
    Ok(Self { pp, nodes })
  }

  // --------------------------------------------------------------------------------------
  // --------------------------------------------------------------------------------------

  /// Get all nodes from given instance
  pub fn get_nodes(&self) -> Vec<TreeNode<G1, G2, C1, C2>> {
    self.nodes.clone()
  }

  /// Get current length of current level
  pub fn get_tree_size(&self) -> usize {
    self.nodes.len()
  }
}

mod tests {
  use super::*;
  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;
  use crate::mapreduce::circuit::{MapReduceArity, MapReduceCircuit};
  use crate::traits::circuit::{StepCircuit, TrivialTestCircuit};
  use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct AverageCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> MapReduceCircuit<F> for AverageCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> MapReduceArity {
      MapReduceArity::new(1, 1)
    }

    fn synthesize_map<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x = z[0].clone();
      Ok(vec![x])
    }
    fn synthesize_reduce<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z_left: &[AllocatedNum<F>],
      z_right: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x_l = &z_left[0];
      let x_r = &z_right[0];
      let mul = x_l.mul(cs.namespace(|| "x_left * x_right"), x_r)?;
      let sum = AllocatedNum::alloc(cs.namespace(|| "x_left + x_right"), || {
        Ok(x_l.get_value().unwrap() + x_r.get_value().unwrap())
      })?;
      cs.enforce(
        || "sum = x_l + x_r",
        |lc| lc + x_l.get_variable() + x_r.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + sum.get_variable(),
      );
      Ok(vec![sum])
    }

    fn output_map(&self, z: &[F]) -> Vec<F> {
      vec![z[0]]
    }
    fn output_reduce(&self, z_left: &[F], z_right: &[F]) -> Vec<F> {
      vec![z_left[0] + z_right[0]]
    }
  }

  use eyre::Result;
  #[test]
  fn test_parallel_combine_two_ivc() -> Result<()> {
    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      AverageCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(AverageCircuit::default(), TrivialTestCircuit::default())?;

    // produce a recursive SNARK
    let leaf_0 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      0,
      vec![<G1 as Group>::Scalar::one()],
      vec![<G2 as Group>::Scalar::one()],
    )?;

    let leaf_1 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      1,
      vec![<G1 as Group>::Scalar::one()],
      vec![<G2 as Group>::Scalar::one()],
    )?;

    let res_2 = leaf_0.reduce(leaf_1);
    assert!(res_2.is_ok());
    Ok(())
  }
}
