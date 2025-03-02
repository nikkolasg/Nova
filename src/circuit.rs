//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a curve in a 2-cycle of curves.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is H(params = H(shape, ck), i, z0, zi, U).
//! Each circuit folds the last invocation of the other into the running instance

use crate::{
  constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    utils::{
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec, le_bits_to_num,
    },
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Group, ROCircuitTrait, ROConstantsCircuit,
  },
  Commitment,
};
use bellpepper::gadgets::Assignment;
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, SynthesisError,
};
use ff::Field;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NovaAugmentedCircuitParams {
  limb_width: usize,
  n_limbs: usize,
  is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
}

impl NovaAugmentedCircuitParams {
  pub const fn new(limb_width: usize, n_limbs: usize, is_primary_circuit: bool) -> Self {
    Self {
      limb_width,
      n_limbs,
      is_primary_circuit,
    }
  }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NovaAugmentedCircuitInputs<G: Group> {
  params: G::Scalar,
  i: G::Base,
  z0: Vec<G::Base>,
  zi: Option<Vec<G::Base>>,
  U: Option<RelaxedR1CSInstance<G>>,
  u: Option<R1CSInstance<G>>,
  T: Option<Commitment<G>>,
}

impl<G: Group> NovaAugmentedCircuitInputs<G> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    params: G::Scalar,
    i: G::Base,
    z0: Vec<G::Base>,
    zi: Option<Vec<G::Base>>,
    U: Option<RelaxedR1CSInstance<G>>,
    u: Option<R1CSInstance<G>>,
    T: Option<Commitment<G>>,
  ) -> Self {
    Self {
      params,
      i,
      z0,
      zi,
      U,
      u,
      T,
    }
  }
}

/// The augmented circuit F' in Nova that includes a step circuit F
/// and the circuit for the verifier in Nova's non-interactive folding scheme
pub struct NovaAugmentedCircuit<'a, G: Group, SC: StepCircuit<G::Base>> {
  params: &'a NovaAugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<G>,
  inputs: Option<NovaAugmentedCircuitInputs<G>>,
  step_circuit: &'a SC, // The function that is applied for each step
}

impl<'a, G: Group, SC: StepCircuit<G::Base>> NovaAugmentedCircuit<'a, G, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    params: &'a NovaAugmentedCircuitParams,
    inputs: Option<NovaAugmentedCircuitInputs<G>>,
    step_circuit: &'a SC,
    ro_consts: ROConstantsCircuit<G>,
  ) -> Self {
    Self {
      params,
      inputs,
      step_circuit,
      ro_consts,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      AllocatedRelaxedR1CSInstance<G>,
      AllocatedR1CSInstance<G>,
      AllocatedPoint<G>,
    ),
    SynthesisError,
  > {
    // Allocate the params
    let params = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "params"),
      self.inputs.as_ref().map(|inputs| inputs.params),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![G::Base::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate the running instance
    let U: AllocatedRelaxedR1CSInstance<G> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate the instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.as_ref().and_then(|inputs| inputs.u.as_ref()),
    )?;

    // Allocate T
    let T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.T.map(|T| T.to_coordinates())),
    )?;

    Ok((params, i, z_0, z_i, U, u, T))
  }

  /// Synthesizes base case and returns the new relaxed `R1CSInstance`
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

  /// Synthesizes non base case and returns the new relaxed `R1CSInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    params: &AllocatedNum<G::Base>,
    i: &AllocatedNum<G::Base>,
    z_0: &[AllocatedNum<G::Base>],
    z_i: &[AllocatedNum<G::Base>],
    U: &AllocatedRelaxedR1CSInstance<G>,
    u: &AllocatedR1CSInstance<G>,
    T: &AllocatedPoint<G>,
    arity: usize,
  ) -> Result<(AllocatedRelaxedR1CSInstance<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * arity,
    );
    ro.absorb(params);
    ro.absorb(i);
    for e in z_0 {
      ro.absorb(e);
    }
    for e in z_i {
      ro.absorb(e);
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      params,
      u,
      T,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((U_fold, check_pass))
  }
}

impl<'a, G: Group, SC: StepCircuit<G::Base>> NovaAugmentedCircuit<'a, G, SC> {
  /// synthesize circuit giving constraint system
  pub fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<G::Base>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (params, i, z_0, z_i, U, u, T) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // Synthesize the circuit for the base case and get the new running instance
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "base case"), u.clone())?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &params,
      &i,
      &z_0,
      &z_i,
      &U,
      &u,
      &T,
      arity,
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
      &Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + G::Base::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;

    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input)?;

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // Compute the new hash H(params, Unew, i+1, z0, z_{i+1})
    let mut ro = G::ROCircuit::new(self.ro_consts, NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * arity);
    ro.absorb(&params);
    ro.absorb(&i_new);
    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_next {
      ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), &hash_bits)?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    u.X1
      .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(z_next)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellpepper::{solver::SatisfyingAssignment, test_shape_cs::TestShapeCS};
  type PastaG1 = pasta_curves::pallas::Point;
  type PastaG2 = pasta_curves::vesta::Point;

  use crate::constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
  use crate::provider;
  use crate::{
    bellpepper::r1cs::{NovaShape, NovaWitness},
    gadgets::utils::scalar_as_base,
    provider::poseidon::PoseidonConstantsCircuit,
    traits::circuit::TrivialCircuit,
  };

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_recursive_circuit_with<G1, G2>(
    primary_params: &NovaAugmentedCircuitParams,
    secondary_params: &NovaAugmentedCircuitParams,
    ro_consts1: ROConstantsCircuit<G2>,
    ro_consts2: ROConstantsCircuit<G1>,
    num_constraints_primary: usize,
    num_constraints_secondary: usize,
  ) where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
    let tc1 = TrivialCircuit::default();
    // Initialize the shape and ck for the primary
    let circuit1: NovaAugmentedCircuit<'_, G2, TrivialCircuit<<G2 as Group>::Base>> =
      NovaAugmentedCircuit::new(primary_params, None, &tc1, ro_consts1.clone());
    let mut cs: TestShapeCS<G1> = TestShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, ck1) = cs.r1cs_shape();
    assert_eq!(cs.num_constraints(), num_constraints_primary);

    let tc2 = TrivialCircuit::default();
    // Initialize the shape and ck for the secondary
    let circuit2: NovaAugmentedCircuit<'_, G1, TrivialCircuit<<G1 as Group>::Base>> =
      NovaAugmentedCircuit::new(secondary_params, None, &tc2, ro_consts2.clone());
    let mut cs: TestShapeCS<G2> = TestShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, ck2) = cs.r1cs_shape();
    assert_eq!(cs.num_constraints(), num_constraints_secondary);

    // Execute the base case for the primary
    let zero1 = <<G2 as Group>::Base as Field>::ZERO;
    let mut cs1: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs1: NovaAugmentedCircuitInputs<G2> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<G1>(zero1), // pass zero for testing
      zero1,
      vec![zero1],
      None,
      None,
      None,
      None,
    );
    let circuit1: NovaAugmentedCircuit<'_, G2, TrivialCircuit<<G2 as Group>::Base>> =
      NovaAugmentedCircuit::new(primary_params, Some(inputs1), &tc1, ro_consts1);
    let _ = circuit1.synthesize(&mut cs1);
    let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &ck1).unwrap();
    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&ck1, &inst1, &witness1).is_ok());

    // Execute the base case for the secondary
    let zero2 = <<G1 as Group>::Base as Field>::ZERO;
    let mut cs2: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs2: NovaAugmentedCircuitInputs<G1> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<G2>(zero2), // pass zero for testing
      zero2,
      vec![zero2],
      None,
      None,
      Some(inst1),
      None,
    );
    let circuit2: NovaAugmentedCircuit<'_, G1, TrivialCircuit<<G1 as Group>::Base>> =
      NovaAugmentedCircuit::new(secondary_params, Some(inputs2), &tc2, ro_consts2);
    let _ = circuit2.synthesize(&mut cs2);
    let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &ck2).unwrap();
    // Make sure that it is satisfiable
    assert!(shape2.is_sat(&ck2, &inst2, &witness2).is_ok());
  }

  #[test]
  fn test_recursive_circuit_pasta() {
    let params1 = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<PastaG2> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<PastaG1> = PoseidonConstantsCircuit::default();

    test_recursive_circuit_with::<PastaG1, PastaG2>(
      &params1, &params2, ro_consts1, ro_consts2, 9815, 10347,
    );
  }

  #[test]
  fn test_recursive_circuit_grumpkin() {
    let params1 = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<provider::bn256_grumpkin::grumpkin::Point> =
      PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<provider::bn256_grumpkin::bn256::Point> =
      PoseidonConstantsCircuit::default();

    test_recursive_circuit_with::<
      provider::bn256_grumpkin::bn256::Point,
      provider::bn256_grumpkin::grumpkin::Point,
    >(&params1, &params2, ro_consts1, ro_consts2, 9983, 10536);
  }

  #[test]
  fn test_recursive_circuit_secp() {
    let params1 = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<provider::secp_secq::secq256k1::Point> =
      PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<provider::secp_secq::secp256k1::Point> =
      PoseidonConstantsCircuit::default();

    test_recursive_circuit_with::<
      provider::secp_secq::secp256k1::Point,
      provider::secp_secq::secq256k1::Point,
    >(&params1, &params2, ro_consts1, ro_consts2, 10262, 10959);
  }
}
