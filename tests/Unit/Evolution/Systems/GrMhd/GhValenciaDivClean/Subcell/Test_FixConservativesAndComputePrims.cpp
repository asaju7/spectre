// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/FixConservativesAndComputePrims.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/MagneticFieldTreatment.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename EquationOfStateType>
void test(const EquationOfStateType& eos) {
  using System = grmhd::GhValenciaDivClean::System;

  // Only use 1 grid point to see that we correctly flagged the point as being
  // fixed. We're really only testing that the mutator calls the correct
  // functions.
  const size_t num_pts = 1;
  tnsr::I<DataVector, 3, Frame::Inertial> subcell_coords{num_pts, 0.0};
  tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric{num_pts, 0.0};
  get<0, 0>(spacetime_metric) = -1.0;
  for (size_t i = 1; i < 4; ++i) {
    spacetime_metric.get(i, i) = 1.0;
  }
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_pts, 0.0};
  tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric{num_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = spacetime_metric.get(i + 1, i + 1);
    inverse_spatial_metric.get(i, i) = 1.0 / spatial_metric.get(i, i);
  }
  const Scalar<DataVector> sqrt_det_spatial_metric{
      sqrt(get(determinant(spatial_metric)))};

  const grmhd::ValenciaDivClean::FixConservatives variable_fixer{
      1.e-7,   1.0e-7,
      1.0e-10, 1.0e-10,
      0.0,     0.0,
      1.e-7,   0.0,
      true,    hydro::MagneticFieldTreatment::AssumeNonZero};
  typename System::variables_tag::type cons_vars{num_pts, 0.0};
  get(get<grmhd::ValenciaDivClean::Tags::TildeD>(cons_vars))[0] = 2.e-12;
  get(get<grmhd::ValenciaDivClean::Tags::TildeYe>(cons_vars))[0] = 2.e-13;
  get(get<grmhd::ValenciaDivClean::Tags::TildeTau>(cons_vars))[0] = 1.e-7;
  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(cons_vars) = spacetime_metric;

  const double cutoff_d_for_inversion = 0.0;
  const double density_when_skipping_inversion = 0.0;
  const double kastaun_max_lorentz = 1.0e4;
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      primitive_from_conservative_options(cutoff_d_for_inversion,
                                          density_when_skipping_inversion,
                                          kastaun_max_lorentz);

  auto box = db::create<db::AddSimpleTags<
      evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
      grmhd::ValenciaDivClean::Tags::VariablesNeededFixing,
      typename System::variables_tag, typename System::primitive_variables_tag,
      ::Tags::VariableFixer<grmhd::ValenciaDivClean::FixConservatives>,
      hydro::Tags::EquationOfState<std::unique_ptr<
          typename EquationsOfState::get_eos_base<EquationOfStateType>>>,
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>>(
      subcell_coords, false, cons_vars,
      typename System::primitive_variables_tag::type{num_pts, 1.0e-4},
      variable_fixer,
      std::unique_ptr<
          typename EquationsOfState::get_eos_base<EquationOfStateType>>{
          std::make_unique<EquationOfStateType>(eos)},
      primitive_from_conservative_options);

  using recovery_schemes = tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>;
  db::mutate_apply<grmhd::GhValenciaDivClean::subcell::
                       FixConservativesAndComputePrims<recovery_schemes>>(
      make_not_null(&box));

  // Verify that the conserved variables were fixed
  CHECK(db::get<grmhd::ValenciaDivClean::Tags::VariablesNeededFixing>(box));

  // Manually do a primitive recovery and see that the values match what's in
  // the DataBox.
  typename System::primitive_variables_tag::type expected_prims{num_pts,
                                                                1.0e-4};
  grmhd::ValenciaDivClean::PrimitiveFromConservative<recovery_schemes, true>::
      apply(
          make_not_null(
              &get<hydro::Tags::RestMassDensity<DataVector>>(expected_prims)),
          make_not_null(
              &get<hydro::Tags::ElectronFraction<DataVector>>(expected_prims)),
          make_not_null(&get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
              expected_prims)),
          make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
              expected_prims)),
          make_not_null(
              &get<hydro::Tags::MagneticField<DataVector, 3>>(expected_prims)),
          make_not_null(&get<hydro::Tags::DivergenceCleaningField<DataVector>>(
              expected_prims)),
          make_not_null(
              &get<hydro::Tags::LorentzFactor<DataVector>>(expected_prims)),
          make_not_null(
              &get<hydro::Tags::Pressure<DataVector>>(expected_prims)),
          make_not_null(
              &get<hydro::Tags::Temperature<DataVector>>(expected_prims)),
          db::get<grmhd::ValenciaDivClean::Tags::TildeD>(box),
          db::get<grmhd::ValenciaDivClean::Tags::TildeYe>(box),
          db::get<grmhd::ValenciaDivClean::Tags::TildeTau>(box),
          db::get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(box),
          db::get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(box),
          db::get<grmhd::ValenciaDivClean::Tags::TildePhi>(box), spatial_metric,
          inverse_spatial_metric, sqrt_det_spatial_metric, eos,
          primitive_from_conservative_options);
  CHECK_VARIABLES_APPROX(db::get<typename System::primitive_variables_tag>(box),
                         expected_prims);
}
}  // namespace
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.FixConsAndComputePrims",
    "[Unit][Evolution]") {
  test(EquationsOfState::PolytropicFluid<true>(100.0, 2.0));
  test(EquationsOfState::Barotropic3D(
      EquationsOfState::PolytropicFluid<true>(100.0, 2.0)));
}
