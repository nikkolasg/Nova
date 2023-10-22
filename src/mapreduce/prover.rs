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
  util_cs::test_cs,
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
    if false {
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
        // that left_end + 1 == right_start (which happens for both cases)
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
    println!("---- MAP primary circuit ----");
    let _ = circuit_primary.clone().synthesize(&mut cs_primary)?;
    if false {
      println!("---- MAP primary TEST circuit ----");
      let mut test_cs_primary = TestConstraintSystem::new();
      let _ = circuit_primary.synthesize(&mut test_cs_primary)?;
      if !test_cs_primary.is_satisfied() {
        println!(
          "primary map unsatisfied: {:?}",
          test_cs_primary.which_is_unsatisfied()
        );
        panic!("unsatisfied primary map");
      }
    }

    let (u_primary, w_primary) =
      cs_primary.r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)?;

    // Transform the R1CS into relaxed R1CS so we can give it to the circuit on the other side of the cycle
    // At first step, u must be folded with an empty instance.
    let w_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &w_primary);
    let u_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck_primary, &pp.r1cs_shape_primary, &u_primary);
    let W_primary = w_primary.clone();
    let U_primary = u_primary.clone();

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
        Some(u_primary.clone()),
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

    println!("---- MAP secondary circuit ----");
    let _ = circuit_secondary.clone().synthesize(&mut cs_secondary);
    let (u_secondary, w_secondary) =
      cs_secondary.r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)?;
    if true {
      println!("---- MAP secondary TEST circuit ----");
      let mut test_cs_secondary = TestConstraintSystem::new();
      let _ = circuit_secondary.synthesize(&mut test_cs_secondary)?;
      if !test_cs_secondary.is_satisfied() {
        println!(
          "secondary map unsatisfied: {:?}",
          test_cs_secondary.which_is_unsatisfied()
        );
        panic!("unsatisfied secondary map");
      }
    }

    // IVC proof of the secondary circuit
    let w_secondary =
      RelaxedR1CSWitness::<G2>::from_r1cs_witness(&pp.r1cs_shape_secondary, &w_secondary);
    let u_secondary = RelaxedR1CSInstance::<G2>::from_r1cs_instance(
      &pp.ck_secondary,
      &pp.r1cs_shape_secondary,
      &u_secondary,
    );
    // At the base case, the running instance for the secondary circuit is null since we only executed
    // one step of the function, so previous steps are "null".
    // On the other hand, for the primary circuit, we executed one step of the function and one folding already.
    //let W_secondary = w_secondary.clone();
    //let U_secondary = u_secondary.clone();
    let W_secondary = RelaxedR1CSWitness::<G2>::default(&pp.r1cs_shape_secondary);
    let U_secondary =
      RelaxedR1CSInstance::<G2>::default(&pp.ck_secondary, &pp.r1cs_shape_secondary);

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

    println!(
      " !! End of MAP step: U_primary.W.x: {:?}",
      U_primary.comm_W.to_coordinates()
    );
    println!(
      " !! End of MAP step: U_secondary.W.x: {:?}",
      U_secondary.comm_W.to_coordinates()
    );

    println!(" !! End of MAP step: u_primary.X0: {:?}", u_primary.X[0]);
    println!(" !! End of MAP step: u_primary.X1: {:?}", u_primary.X[1]);
    println!(" !! End of MAP step: u_primary.X2: {:?}", u_primary.X[2]);
    println!(
      " !! End of MAP step: u_secondary.X0: {:?}",
      u_secondary.X[0]
    );
    println!(
      " !! End of MAP step: u_secondary.X1: {:?}",
      u_secondary.X[1]
    );

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
    // Since there are independent leaves, there is no connection between two ranges
    // For example, [0,1] [1,2] are independently proving the computation, as two independent
    // IVC "lines". When we merge, we merge the indices to [0,2].
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
    //println!(
    //  "[+] PROVER: secondary U_fold {:?} with right relaxed: {:?}, gives {:?}",
    //  left_U_secondary.comm_W.to_coordinates(),
    //  right_U_secondary.comm_W.to_coordinates(),
    //  U_secondary.comm_W.to_coordinates(),
    //);

    // Next we construct a proof of this folding and of the invocation of F

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

    println!("----- MERGE primary circuit ----");
    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let _ = circuit_primary.clone().synthesize(&mut cs_primary);
    if true {
      println!("----- MERGE primary TEST circuit ----");
      let mut test_cs_primary = TestConstraintSystem::new();
      let _ = circuit_primary.synthesize(&mut test_cs_primary)?;
      println!("\t--> Are there unsatisfied constraints ?");
      println!("\t{:?}", test_cs_primary.which_is_unsatisfied());
      if !test_cs_primary.is_satisfied() {
        panic!("unsatisfied merge primary circuit");
      }
    }

    let (u_primary, w_primary) =
      cs_primary.r1cs_instance_and_witness(&self.pp.r1cs_shape_primary, &self.pp.ck_primary)?;

    let u_primary = RelaxedR1CSInstance::from_r1cs_instance(
      &self.pp.ck_primary,
      &self.pp.r1cs_shape_primary,
      &u_primary,
    );
    let w_primary = RelaxedR1CSWitness::from_r1cs_witness(&self.pp.r1cs_shape_primary, &w_primary);

    println!("----- MERGE secondary circuit ----");
    // we want to fold
    // * L_i and l_i (just generated) into L_i+1 (proving F^(i)(z0) = zi)
    // * then L_i+1 and R_i+1 (from previous iteration) into U_i+1
    // We do one less folding because the first circuit only "output one" fresh instance, not two.
    // Now we fold the instances of the primary proof
    let (nifs_left_primary, (left_U_primary, left_W_primary)) = NIFS::prove(
      &self.pp.ck_primary,
      &self.pp.ro_consts_primary,
      &self.pp.r1cs_shape_primary,
      &left.U_primary,
      &left.W_primary,
      // we must give the one just generated above to "make the link" - left.u_primary has already been
      // given in the map step.
      &u_primary,
      &w_primary,
      false,
    )?;
    println!(
      "[+] PROVER : (2 step) MERGE  left_U_primary {:?} with previous computed u_primary {:?} gives {:?}",
      left.U_primary.comm_W.to_coordinates(),
      u_primary.comm_W.to_coordinates(),
      left_U_primary.comm_W.to_coordinates()
    );
    println!(
      "\t - left_U.commW {:?} & u.commW {:?} ",
      left.U_primary.comm_W.to_coordinates(),
      u_primary.comm_W.to_coordinates()
    );
    println!(
      "\t - commT {:?}",
      Commitment::<G1>::decompress(&nifs_left_primary.comm_T)?.to_coordinates()
    );
    println!(
      "\t - left_U.u {:?} & u.u {:?} ",
      left.U_primary.u, u_primary.u
    );
    println!(
      "\t - left_U.X0 {:?} & u.X0 {:?} ",
      left.U_primary.X[0], u_primary.X[0]
    );
    println!(
      "\t - left_U.X1 {:?} & u.X1 {:?} ",
      left.U_primary.X[1], u_primary.X[1]
    );
    println!(
      "\t - left_U.X2 {:?} & u.X2 {:?} ",
      left.U_primary.X[2], u_primary.X[2]
    );
    let (nifs_primary, (U_primary, W_primary)) = NIFS::prove(
      &self.pp.ck_primary,
      &self.pp.ro_consts_primary,
      &self.pp.r1cs_shape_primary,
      &left_U_primary,
      &left_W_primary,
      &right.U_primary,
      &right.W_primary,
      true,
    )?;
    println!(
      "[+] PROVER : MERGE SECONDARY U_fold {:?} with right relaxed {:?} gives {:?}",
      left_U_primary.comm_W.to_coordinates(),
      right.U_primary.comm_W.to_coordinates(),
      U_primary.comm_W.to_coordinates()
    );

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
        Some(left.U_primary), // proves correct exec of F^(i)(z0)=zi
        // NOTE how we take the instance just generated here !
        // TODO: We actually should not need to clone since we don't even nee
        Some(u_primary.clone()), // proves correct exec of F(zi,ri) = zi+1 (proves exec of F & the merge)
        Some(right.U_primary),   // proves correct exec of F^(i)(r0)=ri
        Some(RelaxedR1CSInstance::default(
          &self.pp.ck_primary,
          &self.pp.r1cs_shape_primary,
        )),
        Some(Commitment::<G1>::decompress(&nifs_left_primary.comm_T)?),
        Some(Commitment::<G1>::default()), // since we don't fold R and r together, no T as well
        Some(Commitment::<G1>::decompress(&nifs_primary.comm_T)?),
      );
    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      self.pp.augmented_circuit_params_secondary.clone(),
      Some(inputs_secondary),
      self.c_secondary.clone(),
      self.pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.clone().synthesize(&mut cs_secondary);
    if true {
      println!("----- MERGE secondary TEST circuit ----");
      let mut test_cs_secondary = TestConstraintSystem::new();
      let _ = circuit_secondary.synthesize(&mut test_cs_secondary)?;
      println!("\t--> Are there unsatisfied constraints ?");
      println!("\t{:?}", test_cs_secondary.which_is_unsatisfied());
      if !test_cs_secondary.is_satisfied() {
        panic!("Unsatisfied secondary circuit");
      }
    }
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&self.pp.r1cs_shape_secondary, &self.pp.ck_secondary)?;

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
    println!("[+] PROVER MERGE output:");
    println!("\t - (i_start,i_end) = ({},{})", i_start, i_end);
    println!("\t - U_primary = {:?}", U_primary.comm_W.to_coordinates());
    println!(
      "\t - U_secondary.W  = {:?}",
      U_secondary.comm_W.to_coordinates()
    );
    println!("\t\t - U_secondary.u: {:?}", U_secondary.u);
    // z_start is technically useless after the map step, because reduce always take
    // the output and do not hash the first values. Nevertheless since we are in a single circuit
    // we have to give it as input always, as the "map" function is always computed.
    // TODO: for Supernova, it probably can be removed.
    let mut z_start_primary = left.z_start_primary;
    let arity1 = self.c_primary.arity();
    let arity2 = self.c_secondary.arity();
    // Compute the output ourselves Reduce(left_node.output,right_node.output)
    let z_end_primary = self
      .c_primary
      .output_reduce(&left.z_end_primary, &right.z_end_primary);
    assert!(
      z_end_primary.len() == arity1.reduce_output(),
      "prover primary inconsistent reduce output lengths: got {}, wanted {}",
      z_end_primary.len(),
      arity1.reduce_output()
    );
    let z_start_secondary = left.z_start_secondary;
    let z_end_secondary = self
      .c_secondary
      .output_reduce(&left.z_end_secondary, &right.z_end_secondary);
    assert!(
      z_end_secondary.len() == arity2.reduce_output(),
      "prover secondary inconsistent reduce output lengths: got {} wanted {}",
      z_end_secondary.len(),
      arity2.reduce_output()
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

  // TODO: currently just check the shape, need to verify hash consistency
  fn verify(&self) -> Result<(), NovaError> {
    self
      .pp
      .r1cs_shape_primary
      .is_sat_relaxed(
        &self.pp.ck_primary,
        &self.data.U_primary,
        &self.data.W_primary,
      )
      .and_then(|_| {
        self.pp.r1cs_shape_primary.is_sat_relaxed(
          &self.pp.ck_primary,
          &self.data.u_primary,
          &self.data.w_primary,
        )
      })
      .and_then(|_| {
        self.pp.r1cs_shape_secondary.is_sat_relaxed(
          &self.pp.ck_secondary,
          &self.data.U_secondary,
          &self.data.W_secondary,
        )
      })
      .and_then(|_| {
        self.pp.r1cs_shape_secondary.is_sat_relaxed(
          &self.pp.ck_secondary,
          &self.data.u_secondary,
          &self.data.w_secondary,
        )
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
      println!("\t --- !!! MERGE ALL: nodes.len() = {:?}", nodes.len());
      // New nodes list will reduce a half each round
      nodes = nodes
        .chunks(2)
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
  type F1 = pasta_curves::pallas::Scalar;
  type G2 = pasta_curves::vesta::Point;
  type F2 = pasta_curves::vesta::Scalar;
  use crate::mapreduce::circuit::{MapReduceArity, MapReduceCircuit};
  use crate::traits::circuit::{StepCircuit, TrivialTestCircuit};
  use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;
  use ff::PrimeField;
  use num_integer::Average;

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
  fn test_parallel_snark() -> Result<()> {
    let pp = PublicParams::<
      G1,
      G2,
      AverageCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(AverageCircuit::default(), TrivialTestCircuit::default())?;
    let n = 5;
    let (z1s, z2s) = (0..n)
      .map(|i| (vec![F1::from(i)], vec![F2::from(i)]))
      .unzip();
    let total = <G1 as Group>::Scalar::from((0..n).sum::<u64>());
    let snark = ParallelSNARK::prove(
      &pp,
      z1s,
      z2s,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
    )?;
    assert!(snark.get_tree_size() == 1);
    snark.get_nodes()[0].verify()?;
    assert!(snark.get_nodes()[0].data.z_end_primary[0] == total);
    Ok(())
  }

  #[test]
  fn test_mapreduce_two_reduce() -> Result<()> {
    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      AverageCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(AverageCircuit::default(), TrivialTestCircuit::default())?;
    let one = <G1 as Group>::Scalar::one();
    let two = one + one;
    let one2 = <G2 as Group>::Scalar::one();
    let two2 = one2 + one2;
    let four2 = two2 + two2;

    println!(" --- FIRST LEAF PROVING ---");
    let leaf_0 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      0,
      vec![one],
      vec![four2],
    )?;
    leaf_0.verify()?;

    println!(" --- SECOND LEAF PROVING ---");
    let leaf_1 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      1,
      vec![two],
      vec![four2],
    )?;

    leaf_1.verify()?;

    // println!(" --- THIRD LEAF PROVING ---");
    let leaf_2 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      2,
      vec![one],
      vec![four2],
    )?;
    leaf_2.verify()?;

    // println!(" --- FOURTH LEAF PROVING ---");
    let leaf_3 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      3,
      vec![two],
      vec![four2],
    )?;
    leaf_3.verify()?;

    println!(" --- MERGE 01 PROVING ---");
    let merged01 = leaf_0.reduce(leaf_1)?;
    merged01.verify()?;
    // println!("\n--- MERGE 23 PROVING ---\n");
    let merged23 = leaf_2.reduce(leaf_3)?;
    merged23.verify()?;

    // println!("\n--- MERGE of MERGE PROVING ---\n");
    let merged0123 = merged01.reduce(merged23)?;
    merged0123.verify()?;
    Ok(())
  }

  #[test]
  fn test_mapreduce_single() -> Result<()> {
    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      AverageCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(AverageCircuit::default(), TrivialTestCircuit::default())?;
    let one = <G1 as Group>::Scalar::one();
    let two = one + one;
    let one2 = <G2 as Group>::Scalar::one();
    let two2 = one2 + one2;
    let four2 = two2 + two2;

    println!(" --- FIRST LEAF PROVING ---");
    let leaf_0 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      0,
      vec![one],
      vec![four2],
    )?;
    leaf_0.verify()?;

    println!(" --- SECOND LEAF PROVING ---");
    let leaf_1 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      1,
      vec![two],
      vec![four2],
    )?;

    leaf_1.verify()?;

    println!(" --- MERGE PROVING ---");
    let merged = leaf_0.reduce(leaf_1)?;
    merged.verify()?;
    Ok(())
  }

  #[test]
  fn test_secondary_fold_mixed() -> Result<(), NovaError> {
    let pp = PublicParams::<
      G1,
      G2,
      AverageCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(AverageCircuit::default(), TrivialTestCircuit::default())?;
    let one = <G1 as Group>::Scalar::one();
    let two = one + one;
    let one2 = <G2 as Group>::Scalar::one();
    let two2 = one2 + one2;
    let four2 = two2 + two2;

    println!(" --- FIRST LEAF PROVING ---");
    let leaf_0 = TreeNode::map_step(
      &pp,
      AverageCircuit::default(),
      TrivialTestCircuit::default(),
      0,
      vec![one],
      vec![four2],
    )?;
    leaf_0.verify().expect("leaf 0 verification failed");
    let (nifs_left_secondary, (left_U_secondary, left_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &leaf_0.data.U_secondary,
      &leaf_0.data.W_secondary,
      &leaf_0.data.u_secondary,
      &leaf_0.data.w_secondary,
      false,
    )?;
    pp.r1cs_shape_secondary
      .is_sat_relaxed(&pp.ck_secondary, &left_U_secondary, &left_W_secondary)
      .expect("first NIFS is not sat");

    // do it again
    let (nifs_left_secondary, (left_U_secondary, left_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &left_U_secondary,
      &left_W_secondary,
      &left_U_secondary,
      &left_W_secondary,
      true,
    )?;
    pp.r1cs_shape_secondary
      .is_sat_relaxed(&pp.ck_secondary, &left_U_secondary, &left_W_secondary)
      .expect("second NIFS is not sat");

    //println!(" --- SECOND LEAF PROVING ---");
    //let leaf_1 = TreeNode::map_step(
    //  &pp,
    //  AverageCircuit::default(),
    //  TrivialTestCircuit::default(),
    //  1,
    //  vec![two],
    //  vec![four2],
    //)?;
    //leaf_1.verify().expect("leaf 1 verification failed");

    // do first folding and verify correctness
    //let (nifs_left_secondary, (left_U_secondary, left_W_secondary)) = NIFS::prove(
    //  &pp.ck_secondary,
    //  &pp.ro_consts_secondary,
    //  &pp.r1cs_shape_secondary,
    //  &leaf_0.data.U_secondary,
    //  &leaf_0.data.W_secondary,
    //  &leaf_0.data.u_secondary,
    //  &leaf_0.data.w_secondary,
    //  false,
    //)?;
    //pp.r1cs_shape_secondary
    //  .is_sat_relaxed(&pp.ck_secondary, &left_U_secondary, &left_W_secondary)
    //  .expect("left secondary instance is not sat");

    //let (nifs_secondary, (U_secondary, W_secondary)) = NIFS::prove(
    //  &pp.ck_secondary,
    //  &pp.ro_consts_secondary,
    //  &pp.r1cs_shape_secondary,
    //  &left_U_secondary,
    //  &left_W_secondary,
    //  &leaf_1.data.U_secondary,
    //  &leaf_1.data.W_secondary,
    //  true,
    //)?;
    //pp.r1cs_shape_secondary
    //  .is_sat_relaxed(&pp.ck_secondary, &U_secondary, &W_secondary)
    //  .expect("secondary instance is not sat");

    Ok(())
  }
}
