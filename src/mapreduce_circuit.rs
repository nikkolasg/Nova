//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is
//! H(params = H(shape, ck), i, z0, zi, U). Each circuit folds the last invocation of
//! the other into the running instance

#![allow(unused)]
use crate::{
  circuit::NovaAugmentedCircuitParams,
  constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    utils::{
      alloc_num_equals, alloc_scalar_as_base, conditionally_select_vec, le_bits_to_num,
      FixedSizeAllocator, ResizeAllocator, VecAllocator,
    },
  },
  r1cs::RelaxedR1CSInstance,
  traits::{
    circuit::{MapReduceArity, MapReduceCircuit, StepCircuit},
    commitment::CommitmentTrait,
    Group, ROCircuitTrait, ROConstantsCircuit,
  },
  Commitment,
};
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    Assignment,
  },
  Circuit, ConstraintSystem, SynthesisError,
};
use ff::Field;
use rayon::iter::empty;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NovaAugmentedParallelCircuitInputs<G: Group> {
  params: G::Scalar, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
  i_start_U: G::Base,
  i_end_U: G::Base,
  i_start_R: G::Base,
  i_end_R: G::Base,
  z_U_start: Vec<G::Base>,
  z_U_end: Vec<G::Base>,
  z_R_start: Vec<G::Base>,
  z_R_end: Vec<G::Base>,
  U: Option<RelaxedR1CSInstance<G>>,
  u: Option<RelaxedR1CSInstance<G>>,
  R: Option<RelaxedR1CSInstance<G>>,
  r: Option<RelaxedR1CSInstance<G>>,
  T_u: Option<Commitment<G>>,
  T_r: Option<Commitment<G>>,
  T_R_U: Option<Commitment<G>>,
}

impl<G: Group> NovaAugmentedParallelCircuitInputs<G> {
  /// Create new inputs/witness for the verification circuit
  #[allow(clippy::too_many_arguments)]
  // Remove when we write the struct implementing the parallel nova instance
  #[allow(unused)]
  pub fn new(
    params: G::Scalar, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
    i_start_U: G::Base,
    i_end_U: G::Base,
    i_start_R: G::Base,
    i_end_R: G::Base,
    z_U_start: Vec<G::Base>,
    z_U_end: Vec<G::Base>,
    z_R_start: Vec<G::Base>,
    z_R_end: Vec<G::Base>,
    U: Option<RelaxedR1CSInstance<G>>,
    u: Option<RelaxedR1CSInstance<G>>,
    R: Option<RelaxedR1CSInstance<G>>,
    r: Option<RelaxedR1CSInstance<G>>,
    T_u: Option<Commitment<G>>,
    T_r: Option<Commitment<G>>,
    T_R_U: Option<Commitment<G>>,
  ) -> Self {
    Self {
      params,
      i_start_U,
      i_end_U,
      i_start_R,
      i_end_R,
      z_U_start,
      z_U_end,
      z_R_start,
      z_R_end,
      U,
      u,
      R,
      r,
      T_u,
      T_r,
      T_R_U,
    }
  }
}

/// The augmented circuit F' in Nova that includes a step circuit F
/// and the circuit for the verifier in Nova's non-interactive folding scheme
pub struct NovaAugmentedParallelCircuit<G: Group, MR: MapReduceCircuit<G::Base>> {
  params: NovaAugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<G>,
  inputs: Option<NovaAugmentedParallelCircuitInputs<G>>,
  mr_circuit: MR, // The function that is applied for each step
}

impl<G: Group, MR: MapReduceCircuit<G::Base>> NovaAugmentedParallelCircuit<G, MR> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  // Remove when we write the struct implementing the parallel nova instance
  #[allow(unused)]
  pub fn new(
    params: NovaAugmentedCircuitParams,
    inputs: Option<NovaAugmentedParallelCircuitInputs<G>>,
    mr_circuit: MR,
    ro_consts: ROConstantsCircuit<G>,
  ) -> Self {
    Self {
      params,
      inputs,
      mr_circuit,
      ro_consts,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    arity: MapReduceArity,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      AllocatedRelaxedR1CSInstance<G>,
      AllocatedR1CSInstance<G>,
      AllocatedRelaxedR1CSInstance<G>,
      AllocatedR1CSInstance<G>,
      AllocatedPoint<G>,
      AllocatedPoint<G>,
      AllocatedPoint<G>,
    ),
    SynthesisError,
  > {
    // Allocate the params
    let params = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "params"),
      self.inputs.get().map_or(None, |inputs| Some(inputs.params)),
    )?;

    // Allocate idexes
    let i_start_U = AllocatedNum::alloc(cs.namespace(|| "i_start_U"), || {
      Ok(self.inputs.get()?.i_start_U)
    })?;
    let i_end_U = AllocatedNum::alloc(cs.namespace(|| "i_end_U"), || {
      Ok(self.inputs.get()?.i_end_U)
    })?;
    let i_start_R = AllocatedNum::alloc(cs.namespace(|| "i_start_R"), || {
      Ok(self.inputs.get()?.i_start_R)
    })?;
    let i_end_R = AllocatedNum::alloc(cs.namespace(|| "i_end_R"), || {
      Ok(self.inputs.get()?.i_end_R)
    })?;

    let zero_field = || <G as Group>::Base::zero();
    let inputs = self.inputs.get()?;
    // Allocate input and output vectors
    let z_U_start = inputs.z_U_start.alloc_fixed_size(
      cs.namespace(|| "z_U_start"),
      arity.total_input(),
      "z_U_start",
    )?;
    //let z_U_start = (0..arity.total_input())
    //  .map(|i| {
    //    AllocatedNum::alloc(cs.namespace(|| format!("z_U_start_{i}")), || {
    //      // we pad to the max arity of the circuit - see MapReduceArity for more details
    //      Ok(self.inputs.get()?.z_U_start.get(i).unwrap_or(&zero_field))
    //    })
    //  })
    //  .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;
    let z_U_end = inputs.z_U_end.alloc_fixed_size(
      cs.namespace(|| "z_U_end"),
      arity.total_output(),
      "z_U_end",
    )?;
    //let z_U_end = (0..arity.total_output())
    //  .map(|i| {
    //    AllocatedNum::alloc(cs.namespace(|| format!("z_U_end_{i}")), || {
    //      Ok(self.inputs.get()?.z_U_end.get(i).unwrap_or(&zero_field))
    //    })
    //  })
    //  .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate z_R_start
    let z_R_start = inputs.z_R_start.alloc_fixed_size(
      cs.namespace(|| "z_R_start"),
      arity.total_output(),
      "z_R_start",
    )?;
    //let z_R_start = (0..arity.total_input())
    //  .map(|i| {
    //    AllocatedNum::alloc(cs.namespace(|| format!("z_R_start_{i}")), || {
    //      Ok(self.inputs.get()?.z_R_start.get(i).unwrap_or(&zero_field))
    //    })
    //  })
    //  .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;
    // Allocate z_R_end

    let z_R_end = inputs.z_R_end.alloc_fixed_size(
      cs.namespace(|| "z_R_end"),
      arity.total_output(),
      "z_R_end",
    )?;
    //let z_R_end = (0..arity.total_output())
    //  .map(|i| {
    //    AllocatedNum::alloc(cs.namespace(|| format!("z_R_end_{i}")), || {
    //      Ok(self.inputs.get()?.z_R_end.get(i).unwrap_or(&zero_field))
    //    })
    //  })
    //  .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate the running instance U
    let U: AllocatedRelaxedR1CSInstance<G> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.U.get().map_or(None, |U| Some(U.clone()))
      }),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate the instance u to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.u.get().map_or(None, |u| Some(u.clone()))
      }),
    )?;

    // Allocate the running instance U
    let R: AllocatedRelaxedR1CSInstance<G> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "Allocate R"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.R.get().map_or(None, |R| Some(R.clone()))
      }),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate the instance r to be folded in
    let r = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance r to fold"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.r.get().map_or(None, |r| Some(r.clone()))
      }),
    )?;

    // Allocate T
    let T_u = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T_u"),
      self.inputs.get().map_or(None, |inputs| {
        inputs
          .T_r
          .get()
          .map_or(None, |T_r| Some(T_r.to_coordinates()))
      }),
    )?;

    let T_r = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T_r"),
      self.inputs.get().map_or(None, |inputs| {
        inputs
          .T_u
          .get()
          .map_or(None, |T_u| Some(T_u.to_coordinates()))
      }),
    )?;

    let T_R_U = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T_R_U"),
      self.inputs.get().map_or(None, |inputs| {
        inputs
          .T_R_U
          .get()
          .map_or(None, |T_R_U| Some(T_R_U.to_coordinates()))
      }),
    )?;

    Ok((
      params, i_start_U, i_end_U, i_start_R, i_end_R, z_U_start, z_U_end, z_R_start, z_R_end, U, u,
      R, r, T_u, T_r, T_R_U,
    ))
  }

  /// Synthesizes base case and returns the new relaxed R1CSInstance
  fn synthesize_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstance<G>,
  ) -> Result<AllocatedRelaxedR1CSInstance<G>, SynthesisError> {
    let U_default: AllocatedRelaxedR1CSInstance<G> = if self.params.is_primary_circuit {
      // The primary circuit just returns the default R1CS instance
      AllocatedRelaxedR1CSInstance::default(
        cs.namespace(|| "Allocate U_default"),
        self.params.limb_width,
        self.params.n_limbs,
      )?
    } else {
      // The secondary circuit returns the incoming R1CS instance
      AllocatedRelaxedR1CSInstance::from_r1cs_instance(
        cs.namespace(|| "Allocate U_default"),
        u,
        self.params.limb_width,
        self.params.n_limbs,
      )?
    };
    Ok(U_default)
  }

  /// Synthesizes non base case and returns the new relaxed R1CSInstance
  /// And a boolean indicating if all checks pass
  #[allow(clippy::too_many_arguments)]
  fn synthesize_non_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    params: AllocatedNum<G::Base>,
    i_start_U: AllocatedNum<G::Base>,
    i_end_U: AllocatedNum<G::Base>,
    i_start_R: AllocatedNum<G::Base>,
    i_end_R: AllocatedNum<G::Base>,
    z_U_start: Vec<AllocatedNum<G::Base>>,
    z_U_end: Vec<AllocatedNum<G::Base>>,
    z_R_start: Vec<AllocatedNum<G::Base>>,
    z_R_end: Vec<AllocatedNum<G::Base>>,
    U: AllocatedRelaxedR1CSInstance<G>,
    u: AllocatedR1CSInstance<G>,
    R: AllocatedRelaxedR1CSInstance<G>,
    r: AllocatedR1CSInstance<G>,
    T_u: AllocatedPoint<G>,
    T_r: AllocatedPoint<G>,
    T_R_U: AllocatedPoint<G>,
    arity: MapReduceArity,
  ) -> Result<(AllocatedRelaxedR1CSInstance<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, i_start_U, i_end_U z_U_start, z_U_end, U)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      // one circuit for map & reduce so we take the max
      NUM_FE_WITHOUT_IO_FOR_CRHF + arity.total_input() + arity.total_output() + 1,
    );
    ro.absorb(params.clone());
    ro.absorb(i_start_U.clone());
    ro.absorb(i_end_U.clone());

    // TODO only digest inputs and outputs consumed, in reduce steps that'd be arity.reduce_input()
    for e in z_U_start.clone() {
      ro.absorb(e);
    }
    for e in z_U_end {
      ro.absorb(e);
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash first"), NUM_HASH_BITS)?;
    let hash_u = le_bits_to_num(cs.namespace(|| "bits to hash first"), hash_bits)?;

    let check_pass_u = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z_u_start, z_u_end)"),
      &u.X0,
      &hash_u,
    )?;

    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      params.clone(),
      u,
      T_u,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Check that r.x[0] = Hash(params, i_start_R, i_end_R, z_R_start, z_R_end, R)
    ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      // one circuit for both map and reduce so we take the max
      NUM_FE_WITHOUT_IO_FOR_CRHF + arity.total_input() + arity.total_output() + 1,
    );
    ro.absorb(params.clone());
    ro.absorb(i_start_R);
    ro.absorb(i_end_R);
    for e in z_R_start.clone() {
      ro.absorb(e);
    }
    for e in z_R_end.clone() {
      ro.absorb(e);
    }
    R.absorb_in_ro(cs.namespace(|| "absorb R"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash second"), NUM_HASH_BITS)?;
    let hash_r = le_bits_to_num(cs.namespace(|| "bits to hash second"), hash_bits)?;
    let check_pass_r = alloc_num_equals(
      cs.namespace(|| "check consistency of r.X[0] with H(params, R, i, z_r_start, z_r_end)"),
      &r.X0,
      &hash_r,
    )?;

    // Run NIFS Verifier
    let R_fold = R.fold_with_r1cs(
      cs.namespace(|| "compute fold of R and r"),
      params.clone(),
      r,
      T_r,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Run NIFS Verifier
    let U_R_fold = U_fold.fold_with_relaxed_r1cs(
      cs.namespace(|| "compute fold of U and R"),
      params,
      R_fold,
      T_R_U,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let hashChecks = AllocatedBit::and(
      cs.namespace(|| "check both hashes are correct"),
      &check_pass_u,
      &check_pass_r,
    )?;

    Ok((U_R_fold, hashChecks))
  }
}

impl<G: Group, MR: MapReduceCircuit<G::Base>> Circuit<<G as Group>::Base>
  for NovaAugmentedParallelCircuit<G, MR>
{
  fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    let arity = self.mr_circuit.arity();

    // Allocate all witnesses
    let (
      params,
      i_start_U,
      i_end_U,
      i_start_R,
      i_end_R,
      z_U_start,
      z_U_end,
      z_R_start,
      z_R_end,
      U,
      u,
      R,
      r,
      T_u,
      T_r,
      T_R_U,
    ) = self.alloc_witness(
      cs.namespace(|| "allocate the circuit witness"),
      arity.clone(),
    )?;

    // Compute variable indicating if this is the base case
    let mut is_base_case = alloc_num_equals(
      cs.namespace(|| "Check if base case as i_start_U == i_end_U"),
      &i_start_U.clone(),
      &i_end_U,
    )?;
    let r_index_equal = alloc_num_equals(
      cs.namespace(|| "In base case i_start_R == i_end_R"),
      &i_start_R,
      &i_end_R,
    )?;
    is_base_case = AllocatedBit::and(
      cs.namespace(|| "i_start_U == i_end_U and i_start_U + 1 == i_end_R"),
      &is_base_case,
      &r_index_equal,
    )?;

    // Synthesize the circuit for the base case and get the new running instance
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "base case"), u.clone())?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      params.clone(),
      i_start_U.clone(),
      i_end_U.clone(),
      i_start_R.clone(),
      i_end_R.clone(),
      z_U_start.clone(),
      z_U_end.clone(),
      z_R_start.clone(),
      z_R_end.clone(),
      U,
      u.clone(),
      R,
      r.clone(),
      T_u,
      T_r,
      T_R_U,
      arity.clone(),
    )?;

    // Either check_non_base_pass=true or we are in the base case
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      &check_non_base_pass,
      &is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    // Compute the U_new
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i_u_end == i_r_start, because we're doing map reduce, the reduce function takes
    // [l_start, l_end] and [r_start, r_end] and we need to make sure we're reducing two consecutive ranges
    // (actually this should not even be needed in the general case as reduce step is associative
    // but let's put it there for the time being for the sake of staying closer to PSE parallel code).
    cs.enforce(
      || "check consecutive range",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_end_U.get_variable() - i_start_R.get_variable(),
    );

    let mut z_map_next = self
      .mr_circuit
      // we know the map is the base case and thus takes the input
      .synthesize_map(&mut cs.namespace(|| "F_map"), &z_U_start)?;
    let mut z_reduce_next = self
        .mr_circuit
        // we know exactly the reduce function takes the outputs of the two children nodes
        .synthesize_reduce(&mut cs.namespace(|| "F_reduce"), &z_U_end, &z_R_end)?;

    if z_map_next.len() != arity.map_output() {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_map_next".to_string(),
      ));
    }

    if z_reduce_next.len() != arity.reduce_output() {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_reduce_next".to_string(),
      ));
    }

    // we put them on the same size because once again we only have one circuit so
    // the output size should be the same regardless of the steps we're doing because
    // we're hashing them
    // TODO: enforce to only hash the relevant field, for this PoC it's ok
    z_reduce_next.alloc_resize(cs.namespace(|| ""), arity.total_output())?;
    z_map_next.alloc_resize(cs.namespace(|| ""), arity.total_output())?;

    // In the base case our output is in the z_map_next and in our normal case it's in z_reduce_next
    let z_output = conditionally_select_vec(
      cs.namespace(|| "select output of F"),
      &z_map_next,
      &z_reduce_next,
      &Boolean::from(is_base_case),
    )?;

    // Compute the new hash H(params, Unew, i_left_start, i_right_end, z_left_start, z_output)
    // i.e. we put the interval [left start i, right end i] and we give as z_0 the most left start value
    let mut ro = G::ROCircuit::new(
      self.ro_consts,
      NUM_FE_WITHOUT_IO_FOR_CRHF + arity.total_output() + arity.total_input() + 1,
    );
    ro.absorb(params);
    ro.absorb(i_start_U.clone());
    ro.absorb(i_end_R.clone());
    for e in z_U_start {
      ro.absorb(e);
    }
    for e in z_output {
      ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;
    u.X1
      .inputize(cs.namespace(|| "Output unmodified hash of the u circuit"))?;
    // r.X1.inputize(cs.namespace(|| "Output unmodified hash of the r circuit"))?;

    Ok(())
  }
}
