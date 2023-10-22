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
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select,
      conditionally_select_vec, le_bits_to_num, FixedSizeAllocator, ResizeAllocator,
    },
  },
  mapreduce::circuit::{MapReduceArity, MapReduceCircuit},
  r1cs::RelaxedR1CSInstance,
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Group, ROCircuitTrait, ROConstantsCircuit,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Clone)]
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

    // Allocate input and output vectors
    // z_U_start is either (a) map input or (b) map output or reduce output (==). Both
    // case have different arity so we allocate for the max of both cases and pad with 0
    // in case we have a smaller arity than the max.
    let z_U_start = AllocatedNum::alloc_fixed_size(
      cs.namespace(|| "z_U_start"),
      arity.total_input(), // makes sure the input size allocation works for both map and reduce
      || self.inputs.get().map(|s| &s.z_U_start),
    )?;
    let z_U_end =
      AllocatedNum::alloc_fixed_size(cs.namespace(|| "z_U_end"), arity.total_output(), || {
        self.inputs.get().map(|s| &s.z_U_end)
      })?;

    // Allocate z_R_start
    let z_R_start =
      AllocatedNum::alloc_fixed_size(cs.namespace(|| "z_R_start"), arity.total_input(), || {
        self.inputs.get().map(|s| &s.z_R_start)
      })?;

    // Allocate z_R_end
    let z_R_end =
      AllocatedNum::alloc_fixed_size(cs.namespace(|| "z_R_end"), arity.total_output(), || {
        self.inputs.get().map(|s| &s.z_R_end)
      })?;

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
          .T_u
          .get()
          .map_or(None, |T_u| Some(T_u.to_coordinates()))
      }),
    )?;

    let T_r = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T_r"),
      self.inputs.get().map_or(None, |inputs| {
        inputs
          .T_r
          .get()
          .map_or(None, |T_r| Some(T_r.to_coordinates()))
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
    is_base_case: &AllocatedBit,
  ) -> Result<(AllocatedRelaxedR1CSInstance<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, i_start_U, i_end_U, z_U_end, U)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      // one circuit for map & reduce so we take the max
      NUM_FE_WITHOUT_IO_FOR_CRHF + arity.total_output() + 1,
    );
    ro.absorb(params.clone());
    ro.absorb(i_start_U.clone());
    ro.absorb(i_end_U.clone());

    // see synthetize method end for more infor as for why it's commented out.
    //for e in z_U_start.clone() {
    //  ro.absorb(e);
    //}
    for e in &z_U_end {
      ro.absorb(e.clone());
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    println!("[+] Hash Consistency Check for U:");
    println!("\t - params: {:?}", params.get_value());
    println!("\t - i_start_U: {:?}", i_start_U.get_value());
    println!("\t - i_end_U: {:?}", i_end_U.get_value());
    println!(
      "\t - z_U_end (len {:?}): {:?}",
      z_U_end.len(),
      z_U_end[0].get_value()
    );
    println!(
      "\t - U.W.X: {:?}, {:?}, {:?}",
      U.W.x.get_value(),
      U.W.y.get_value(),
      U.W.is_infinity.get_value()
    );
    println!(
      "\t - U.E.X: {:?}, {:?} {:?}",
      U.E.x.get_value(),
      U.E.y.get_value(),
      U.E.is_infinity.get_value()
    );
    println!("\t - U.u: {:?}", U.u.value);
    println!("\t - U.X0: {:?}", U.X0.value);
    println!("\t - U.X1: {:?}", U.X1.value);
    println!("\t - U.X2 {:?}", U.X2.value);

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash first"), NUM_HASH_BITS)?;
    let hash_u = le_bits_to_num(cs.namespace(|| "bits to hash first"), hash_bits)?;
    println!("  ==> Computed = {:?}", hash_u.get_value());
    println!("  ==> Expected = {:?}", u.X0.get_value());
    let check_pass_u = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z_u_end)"),
      &u.X0,
      &hash_u,
    )?;

    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      params.clone(),
      u.clone(), // ARGH. All these function should take a reference.
      T_u.clone(),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    println!(
      "[+] CIRCUIT: MERGE (a) left_U {:?} with previous computed u {:?} gives {:?}",
      U.W.x.get_value(),
      u.W.x.get_value(),
      U_fold.W.x.get_value()
    );
    println!(
      "\t - left_U.u {:?} & Unew.u {:?}",
      U.u.value, U_fold.u.value
    );

    // CHECK if we are in secondary circuit: if we are, then the folding operates a little bit differently.
    // In first circuit, we operate  as per the book.
    // In second circuit, we don't take "r" as it is null, but we take u.X1, because it comes from the previous circuit
    // on the first curve that already "merged" right and left.
    // To check if we are on second circuit,we enforce that the right instance is null. We here simply check if
    // the first input is 0 or not (assuming we're not in base case).
    let zero_num = &alloc_zero(cs.namespace(|| "alloczero"))?;
    let is_r_instance_zero = alloc_num_equals(
      cs.namespace(|| "check right instance zero"),
      &r.X0,
      &zero_num,
    )?;
    let is_nonbase_secondary = AllocatedBit::and_not(
      cs.namespace(|| "check if we are in non base case & secondary circuit"),
      &is_r_instance_zero,
      &is_base_case,
    )?;
    println!(
      "[+] is SECONDARY Non Base circuit ? {:?} ",
      is_nonbase_secondary.get_value()
    );
    let right_io = conditionally_select(
      cs.namespace(|| "compute U_new"),
      &u.X1,
      &r.X0,
      &Boolean::from(is_nonbase_secondary.clone()),
    )?;

    // Check that r.x[0] = Hash(params, i_start_R, i_end_R, z_R_end, R)
    ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      // one circuit for both map and reduce so we take the max
      NUM_FE_WITHOUT_IO_FOR_CRHF + arity.total_output() + 1,
    );
    ro.absorb(params.clone());
    ro.absorb(i_start_R);
    ro.absorb(i_end_R);
    // See synthetize method for more info as to why this is removed.
    //for e in z_R_start.clone() {
    //  ro.absorb(e);
    //}
    for e in z_R_end.clone() {
      ro.absorb(e);
    }
    R.absorb_in_ro(cs.namespace(|| "absorb R"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash second"), NUM_HASH_BITS)?;
    let hash_r = le_bits_to_num(cs.namespace(|| "bits to hash second"), hash_bits)?;
    println!("NonBaseCase: check Hash_r= {:?}", hash_r.get_value());
    let check_pass_r = alloc_num_equals(
      cs.namespace(|| {
        "check consistency of right instance with H(params, R, i, z_r_start, z_r_end)"
      }),
      &right_io,
      &hash_r,
    )?;

    // Run NIFS Verifier
    // Note in secondary circuit, we don't care about the result of this function.
    let R_fold = R.fold_with_r1cs(
      cs.namespace(|| "compute fold of R and r"),
      params.clone(),
      r.clone(),
      T_r,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    println!(
      "[+] CIRCUIT: MERGE (b) R {:?} with previous computed r {:?} gives {:?}",
      R.W.x.get_value(),
      r.W.x.get_value(),
      R_fold.W.x.get_value()
    );

    println!("\t - R.u {:?} & Unew.u {:?}", U.u.value, U_fold.u.value);

    // In first circuit, we fold U_i+1 and R_i+1
    // In second circuit, we fold U_i+1 and R_i <-- this is because U_i+1 already
    // attests to the validity of the merge function on the first circuit already.
    // Note the prover make sure it gives the right T_R_U corresponding to the right
    // chosen relaxed instance.
    let right_relaxed = R.conditionally_select(
      cs.namespace(|| "choosing right_relaxed"),
      R_fold,
      &Boolean::from(is_nonbase_secondary),
    )?;
    // Run NIFS Verifier
    let U_R_fold = U_fold.fold_with_relaxed_r1cs(
      cs.namespace(|| "compute fold of U and R"),
      params,
      right_relaxed.clone(),
      T_R_U.clone(),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    println!(
      "[+] CIRCUIT: MERGE (c) U_fold {:?} with right relaxed: {:?}, gives {:?}",
      U_fold.W.x.get_value(),
      right_relaxed.W.x.get_value(),
      U_R_fold.W.x.get_value(),
    );
    println!(
      "\t - U_fold.u {:?}\n\t - R_fold {:?}\n\t - Unew.u {:?}",
      U_fold.u.value, right_relaxed.u.value, U_R_fold.u.value
    );

    let hashChecks = AllocatedBit::and(
      cs.namespace(|| "check both hashes are correct"),
      &check_pass_u,
      &check_pass_r,
    )?;
    println!(
      "NonBaseCase: CheckPassU: {:?}, CheckPassR: {:?}, HashChecks: {:?}",
      check_pass_u.get_value(),
      check_pass_r.get_value(),
      hashChecks.get_value()
    );
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
    // Base case description check in Nova paper for F'
    let left_io_equal = alloc_num_equals(
      cs.namespace(|| "Check if base case as i_start_U == i_end_U"),
      &i_start_U.clone(),
      &i_end_U,
    )?;
    // even though we don't use anything from the right side in base case, it's important
    // to check that they're equal. Later in the circuit, we check that i_end_U + 1 == i_start R
    // so we also need to enforce that i_end_R == i_start_R in base case otherwise the resulting range
    // [i_start_U, i_end_R] could be anything because i_end_R would not be constrained.
    let right_io_equal = alloc_num_equals(
      cs.namespace(|| "In base case i_start_R == i_end_R"),
      &i_start_R,
      &i_end_R,
    )?;
    let is_base_case = AllocatedBit::and(
      cs.namespace(|| "i_start_U == i_end_U and i_start=R == i_end_R"),
      &left_io_equal,
      &right_io_equal,
    )?;
    println!("[+] is BASE CASE ? {:?}", is_base_case.get_value());

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
      &is_base_case,
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
    // Compute i_U_end + 1 for base case
    let i_end_U_base = AllocatedNum::alloc(cs.namespace(|| "i_U_end + 1"), || {
      Ok(*i_end_U.get_value().get()? + G::Base::one())
    })?;
    cs.enforce(
      || "check new computed i_end_U + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_end_U_base.get_variable() - CS::one() - i_end_U.get_variable(),
    );
    let i_end_U_new = conditionally_select(
      cs.namespace(|| "choose i_end_U in base vs nonbase case"),
      &i_end_U_base,
      &i_end_U,
      &Boolean::from(is_base_case.clone()),
    )?;
    // Then check the range equation to make sure the right instance starts at the same index than the left instance
    // is ending
    // NOTE: this is in theory strictly not neeeded as reduce step is associative but in things get messy in the constraints
    // world so we force this consecutive range.
    // Example:
    //  * in base case, we initially have [0,0] [1,1]. Then i_U_end_new = 1 so condition checks out, output range is [0,1]
    //  * in non base case, we have for example [0,1] [1,2]. Then i_U_end_new = 1 (no adding via condition above).
    cs.enforce(
      || "check consecutive range",
      |lc| lc + i_end_U_new.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + i_start_R.get_variable(),
    );

    let mut z_map_next = self
      .mr_circuit
      // we know the map is the base case and thus takes the input
      // We could also take z_U_end but in the base case, it's just simpler to think that we take the input to 
      // the map function.
      // NOTE: here we only take the right amount of input wires, since we allocated for the max between
      // map and reduce !
      .synthesize_map(&mut cs.namespace(|| "F_map"), &z_U_start[..arity.map_input()])?;
    // NOTE: here we take the size of the output of the map function since it's the same as the size
    // of the reduce function
    let mut z_reduce_next = self.mr_circuit.synthesize_reduce(
      &mut cs.namespace(|| "F_reduce"),
      &z_U_end[..arity.reduce_input()],
      &z_R_end[..arity.reduce_input()],
    )?;

    if z_map_next.len() != arity.map_output() {
      return Err(SynthesisError::IncompatibleLengthVector(format!(
        "z_map_next: got {}, wanted {}",
        z_map_next.len(),
        arity.map_output()
      )));
    }

    if z_reduce_next.len() != arity.reduce_output() {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_reduce_next".to_string(),
      ));
    }

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
      NUM_FE_WITHOUT_IO_FOR_CRHF + arity.total_output() + 1,
    );
    ro.absorb(params.clone());
    ro.absorb(i_start_U.clone());
    ro.absorb(i_end_R.clone());
    // NOTE: this is commented out in the case of mapreduce or general PCD, because
    // the verifier can not have a public input of the size of the input set (remember
    // the values are independent!).
    // As a consequence it means the verifier does not know what are the input
    // values of the prover unlike IVC. In our case, we actually check the
    // validity of the values inside the map/reduce computation, for example,
    // to make sure they belong in a Merkle Tree.
    // One could directly do this check here, by hashing z_U_start and z_U_start and
    // input the result into this output hash, thereby "committing" to the input values.
    // This would force the verifier to do this computation himself which is not desirable
    // at all even though it could be done outside of the circuit. Our approach is to
    // verify merkle proofs inside the computation and recreate the root of the merkle tree
    // as output so the verifier can easily know whether the values corresponds to the tree he
    // knows about.
    //for e in z_U_start {
    //  ro.absorb(e);
    //}
    for e in &z_output {
      ro.absorb(e.clone());
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;

    println!("[+] Hash Output X2:");
    println!("\t - params: {:?}", params.get_value());
    println!("\t - i_start_U: {:?}", i_start_U.get_value());
    println!("\t - i_end_R: {:?}", i_end_R.get_value());
    println!(
      "\t - z_output (len {:?}): {:?}",
      z_output.len(),
      z_output[0].get_value()
    );
    println!(
      "\t - U.W.X: {:?}, {:?}, {:?}",
      Unew.W.x.get_value(),
      Unew.W.y.get_value(),
      Unew.W.is_infinity.get_value()
    );
    println!(
      "\t - U.E.X: {:?}, {:?} {:?}",
      Unew.E.x.get_value(),
      Unew.E.y.get_value(),
      Unew.E.is_infinity.get_value()
    );
    println!("\t - Unew.u: {:?}", Unew.u.value);
    println!("\t - Unew.X0: {:?}", Unew.X0.value);
    println!("\t - Unew.X1: {:?}", Unew.X1.value);
    println!("\t - Unew.X2: {:?}", Unew.X2.value);

    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;
    println!(" ==> X2 H = {:?}", hash.get_value());
    println!(" ==> X0 H = {:?}", u.X2.get_value());
    println!(" ==> X1 H = {:?}", r.X2.get_value());

    u.X2
      .inputize(cs.namespace(|| "Output unmodified hash of the u circuit"))?;
    // In non base case secondary circuit, this will be null
    r.X2
      .inputize(cs.namespace(|| "Output unmodified hash of the r circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(())
  }
}

//pub struct ROCircuitLog<F: Field, R: ROCircuitTrait<F>> {
//  rocircuit: R,
//}

//impl<F,R> ROCircuitTrait<F> for ROCircuitLog<F,R> where
//  Scalar: PrimeField + PrimeFieldBits + Serialize + for<'de> Deserialize<'de>,
//  R: ROCircuitTrait<F>,
//  {
//    type Constants = R::Constants;
//    fn new(constants: Self::Constants, num_absorbs: usize) -> Self {
//      Self {
//        R::new(constants, num_absorbs)
//      }
//    }
//    fn absorb(&mut self, e: AllocatedNum<F>) {
//        print!("\t - Absorb: {:?}",e);
//        self.rocircuit.absorb(e);
//    }
//    fn squeeze<CS>(&mut self, cs: CS, num_bits: usize) -> Result<Vec<AllocatedBit>, SynthesisError>
//      where
//        CS: ConstraintSystem<F> {
//        self.rocircuit.squeeze(cs, num_bits)
//    }
//}
