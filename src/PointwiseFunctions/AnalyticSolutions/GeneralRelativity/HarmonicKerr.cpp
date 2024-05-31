// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicKerr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr::Solutions {

HarmonicKerr::HarmonicKerr(
    const double mass, const std::array<double, volume_dim>& dimensionless_spin,
    const std::array<double, volume_dim>& center,
    const Options::Context& context)
    : mass_(mass), dimensionless_spin_(dimensionless_spin), center_(center) {
  const double spin_magnitude = magnitude(dimensionless_spin_);
  if (spin_magnitude > 1.) {
    PARSE_ERROR(context, "Spin magnitude must be < 1. Given spin: "
                             << dimensionless_spin_ << " with magnitude "
                             << spin_magnitude);
  }
  if (mass_ < 0.) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

HarmonicKerr::HarmonicKerr(CkMigrateMessage* /*msg*/) {}

void HarmonicKerr::pup(PUP::er& p) {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

template <typename DataType, typename Frame>
HarmonicKerr::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const HarmonicKerr& solution, const tnsr::I<DataType, 3, Frame>& x)
    : solution_(solution), x_(x) {}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_minus_center<DataType, Frame> /*meta*/) const {
  *x_minus_center = x_;
  for (size_t i = 0; i < 3; ++i) {
    x_minus_center->get(i) -= gsl::at(solution_.center(), i);
  }
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> spin_sq,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::spin_sq<DataType> /*meta*/) const {
  get(*spin_sq) = square(gsl::at(solution_.dimensionless_spin(), 0)) +
                  square(gsl::at(solution_.dimensionless_spin(), 1)) +
                  square(gsl::at(solution_.dimensionless_spin(), 2));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> spin_mag,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::spin_mag<DataType> /*meta*/) const {
  const auto& spin_sq =
      cache->get_var(*this, internal_tags::spin_sq<DataType>{});
  get(*spin_mag) = sqrt(get(spin_sq));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> xc_sq_minus_a_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::xc_sq_minus_a_sq<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& xc_sq = square(get<0>(x_minus_center)) +
                      square(get<1>(x_minus_center)) +
                      square(get<2>(x_minus_center));

  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});

  get(*xc_sq_minus_a_sq) = xc_sq - get(a_sq);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_minus_mass_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_minus_mass_sq<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& xc_sq = square(get<0>(x_minus_center)) +
                      square(get<1>(x_minus_center)) +
                      square(get<2>(x_minus_center));

  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});
  const auto& xc_sq_minus_a_sq =
      cache->get_var(*this, internal_tags::xc_sq_minus_a_sq<DataType>{});

  const auto& a_dot_xc = gsl::at(solution_.dimensionless_spin(), 2) *
                         get<2>(x_minus_center);  // only allowing spin in z

  DataType diff = xc_sq - get(a_sq);

  if (min(diff) >= 0.) {
    get(*r_minus_mass_sq) =
        0.5 * (get(xc_sq_minus_a_sq) +
               sqrt(square(get(xc_sq_minus_a_sq)) + 4 * square(a_dot_xc)));
  } else {
    get(*r_minus_mass_sq) =
        2 * square(a_dot_xc) /
        (-1 * get(xc_sq_minus_a_sq) +
         sqrt(square(get(xc_sq_minus_a_sq)) + 4 * square(a_dot_xc)));
  }
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const {
  const auto& r_minus_mass_sq =
      cache->get_var(*this, internal_tags::r_minus_mass_sq<DataType>{});

  get(*r) = sqrt(get(r_minus_mass_sq)) + solution_.mass();
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_sq<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});

  get(*r_sq) = square(get(r));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> theta,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::theta<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});
  const auto& r_minus_mass = get(r) - solution_.mass();

  get(*theta) = acos(get<2>(x_minus_center) / (r_minus_mass));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> phi,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::phi<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});
  const auto& r_minus_mass = get(r) - solution_.mass();
  const auto& a = cache->get_var(*this, internal_tags::spin_mag<DataType>{});

  const auto& num = (get<0>(x_minus_center) + get<1>(x_minus_center)) *
                        (r_minus_mass - get(a)) -
                    (get<0>(x_minus_center) - get<1>(x_minus_center)) *
                        (r_minus_mass + get(a));
  const auto& den = (get<0>(x_minus_center) + get<1>(x_minus_center)) *
                        (r_minus_mass + get(a)) +
                    (get<0>(x_minus_center) - get<1>(x_minus_center)) *
                        (r_minus_mass - get(a));

  get(*phi) = atan2(num, den);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> rho_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::rho_sq<DataType> /*meta*/) const {
  const auto& r_sq = cache->get_var(*this, internal_tags::r_sq<DataType>{});
  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});
  const auto& theta = cache->get_var(*this, internal_tags::theta<DataType>{});

  get(*rho_sq) = get(r_sq) + get(a_sq) * cos(get(theta));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> inverse_rho_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::inverse_rho_sq<DataType> /*meta*/) const {
  const auto& rho_sq = cache->get_var(*this, internal_tags::rho_sq<DataType>{});

  get(*inverse_rho_sq) = 1. / get(rho_sq);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_plus_horizon,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_plus_horizon<DataType> /*meta*/) const {
  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});

  get(*r_plus_horizon) =
      solution_.mass() + sqrt(square(solution_.mass()) - get(a_sq));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_minus_horizon,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_minus_horizon<DataType> /*meta*/) const {
  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});

  get(*r_minus_horizon) =
      solution_.mass() - sqrt(square(solution_.mass()) - get(a_sq));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> capital_delta,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::capital_delta<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});
  const auto& r_sq = cache->get_var(*this, internal_tags::r_sq<DataType>{});
  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});

  get(*capital_delta) = get(r_sq) - 2. * solution_.mass() * get(r) + get(a_sq);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_plus_rplus_over_r_minus_rminus,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_plus_rplus_over_r_minus_rminus<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});
  const auto& r_plus_horizon =
      cache->get_var(*this, internal_tags::r_plus_horizon<DataType>{});
  const auto& r_minus_horizon =
      cache->get_var(*this, internal_tags::r_minus_horizon<DataType>{});

  get(*r_plus_rplus_over_r_minus_rminus) =
      (get(r) + get(r_plus_horizon)) / (get(r) - get(r_minus_horizon));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_mass_radius_over_rho_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_mass_radius_over_rho_sq<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});
  const auto& inverse_rho_sq =
      cache->get_var(*this, internal_tags::inverse_rho_sq<DataType>{});

  get(*two_mass_radius_over_rho_sq) =
      (2. * solution_.mass() * get(r)) * (get(inverse_rho_sq));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_sq_plus_a_sq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_sq_plus_a_sq<DataType> /*meta*/) const {
  const auto& r_sq = cache->get_var(*this, internal_tags::r_sq<DataType>{});
  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});

  get(*r_sq_plus_a_sq) = get(r_sq) + get(a_sq);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*>
        spatial_metric_harm_slicing,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::spatial_metric_harm_slicing<DataType, Frame> /*meta*/)
    const {
  const auto& r_plus_rplus_over_r_minus_rminus = cache->get_var(
      *this, internal_tags::r_plus_rplus_over_r_minus_rminus<DataType>{});
  const auto& two_mass_radius_over_rho_sq = cache->get_var(
      *this, internal_tags::two_mass_radius_over_rho_sq<DataType>{});
  const auto& r_sq_plus_a_sq =
      cache->get_var(*this, internal_tags::r_sq_plus_a_sq<DataType>{});
  const auto& rho_sq = cache->get_var(*this, internal_tags::rho_sq<DataType>{});
  const auto& inverse_rho_sq =
      cache->get_var(*this, internal_tags::inverse_rho_sq<DataType>{});
  const auto& capital_delta =
      cache->get_var(*this, internal_tags::capital_delta<DataType>{});
  const auto& a_sq = cache->get_var(*this, internal_tags::spin_sq<DataType>{});
  const auto& a = cache->get_var(*this, internal_tags::spin_mag<DataType>{});
  const auto& theta = cache->get_var(*this, internal_tags::theta<DataType>{});

  // intermediates
  const auto& sin_theta_sq = square(sin(get(theta)));
  const auto& gamma_rr_bracket = 2. - ((1. - get(two_mass_radius_over_rho_sq)) *
                                       get(r_plus_rplus_over_r_minus_rminus));
  const auto& gamma_rphi_parens =
      get(two_mass_radius_over_rho_sq) * get(r_plus_rplus_over_r_minus_rminus) +
      1.;
  const auto& gamma_phiphi_bracket =
      square(get(r_sq_plus_a_sq)) -
      (get(capital_delta) * get(a_sq) * sin_theta_sq);

  spatial_metric_harm_slicing->get(0, 0) =
      gamma_rr_bracket * get(r_plus_rplus_over_r_minus_rminus);
  spatial_metric_harm_slicing->get(0, 1) = 0.;
  spatial_metric_harm_slicing->get(0, 2) =
      gamma_rphi_parens * (-1. * get(a) * sin_theta_sq);

  spatial_metric_harm_slicing->get(1, 0) = 0.;
  spatial_metric_harm_slicing->get(1, 1) = get(rho_sq);
  spatial_metric_harm_slicing->get(1, 2) = 0.;

  spatial_metric_harm_slicing->get(2, 0) =
      gamma_rphi_parens * (-1. * get(a) * sin_theta_sq);
  spatial_metric_harm_slicing->get(2, 1) = 0.;
  spatial_metric_harm_slicing->get(2, 2) =
      gamma_phiphi_bracket * get(inverse_rho_sq) * sin_theta_sq;
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> one_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::one_over_r<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});

  get(*one_over_r) = 1. / get(r);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::x_over_r<DataType, Frame> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType>{});
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});

  tenex::evaluate<ti::I>(x_over_r, x_minus_center(ti::I) * one_over_r());
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> m_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::m_over_r<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType>{});

  get(*m_over_r) = solution_.mass() * get(one_over_r);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> sqrt_f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::sqrt_f_0<DataType> /*meta*/) const {
  const auto& m_over_r =
      cache->get_var(*this, internal_tags::m_over_r<DataType>{});

  get(*sqrt_f_0) = 1. + get(m_over_r);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_0<DataType> /*meta*/) const {
  const auto& sqrt_f_0 =
      cache->get_var(*this, internal_tags::sqrt_f_0<DataType>{});

  get(*f_0) = square(get(sqrt_f_0));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_m_over_m_plus_r<DataType> /*meta*/) const {
  const auto& r = cache->get_var(*this, internal_tags::r<DataType>{});

  get(*two_m_over_m_plus_r) =
      2. * solution_.mass() / (solution_.mass() + get(r));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_m_over_m_plus_r_squared<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r =
      cache->get_var(*this, internal_tags::two_m_over_m_plus_r<DataType>{});

  get(*two_m_over_m_plus_r_squared) = square(get(two_m_over_m_plus_r));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> two_m_over_m_plus_r_cubed,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::two_m_over_m_plus_r_cubed<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r =
      cache->get_var(*this, internal_tags::two_m_over_m_plus_r<DataType>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType>{});

  get(*two_m_over_m_plus_r_cubed) =
      get(two_m_over_m_plus_r) * get(two_m_over_m_plus_r_squared);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> spatial_metric_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::spatial_metric_rr<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r =
      cache->get_var(*this, internal_tags::two_m_over_m_plus_r<DataType>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType>{});

  get(*spatial_metric_rr) = 1. + get(two_m_over_m_plus_r) +
                            get(two_m_over_m_plus_r_squared) +
                            get(two_m_over_m_plus_r_cubed);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> one_over_spatial_metric_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::one_over_spatial_metric_rr<DataType> /*meta*/) const {
  const auto& spatial_metric_rr =
      cache->get_var(*this, internal_tags::spatial_metric_rr<DataType>{});

  get(*one_over_spatial_metric_rr) = 1. / get(spatial_metric_rr);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> spatial_metric_rr_minus_f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::spatial_metric_rr_minus_f_0<DataType> /*meta*/) const {
  const auto& spatial_metric_rr =
      cache->get_var(*this, internal_tags::spatial_metric_rr<DataType>{});
  const auto& f_0 = cache->get_var(*this, internal_tags::f_0<DataType>{});

  get(*spatial_metric_rr_minus_f_0) = get(spatial_metric_rr) - get(f_0);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> d_spatial_metric_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_spatial_metric_rr<DataType> /*meta*/) const {
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType>{});

  get(*d_spatial_metric_rr) =
      (0.5 * get(two_m_over_m_plus_r_squared) + get(two_m_over_m_plus_r_cubed) +
       1.5 * square(get(two_m_over_m_plus_r_squared))) /
      (-solution_.mass());
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> d_f_0,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_f_0<DataType> /*meta*/) const {
  const auto& sqrt_f_0 =
      cache->get_var(*this, internal_tags::sqrt_f_0<DataType>{});
  const auto& m_over_r =
      cache->get_var(*this, internal_tags::m_over_r<DataType>{});
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType>{});

  get(*d_f_0) = -2. * get(sqrt_f_0) * get(m_over_r) * get(one_over_r);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> d_f_0_times_x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::d_f_0_times_x_over_r<DataType, Frame> /*meta*/) const {
  const auto& d_f_0 = cache->get_var(*this, internal_tags::d_f_0<DataType>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  get<0>(*d_f_0_times_x_over_r) = get(d_f_0) * get<0>(x_over_r);
  get<1>(*d_f_0_times_x_over_r) = get(d_f_0) * get<1>(x_over_r);
  get<2>(*d_f_0_times_x_over_r) = get(d_f_0) * get<2>(x_over_r);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_1<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType>{});
  const auto& spatial_metric_rr =
      cache->get_var(*this, internal_tags::spatial_metric_rr<DataType>{});
  const auto& f_0 = cache->get_var(*this, internal_tags::f_0<DataType>{});

  get(*f_1) = get(one_over_r) * (get(spatial_metric_rr) - get(f_0));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> f_1_times_x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_1_times_x_over_r<DataType, Frame> /*meta*/) const {
  const auto& f_1 = cache->get_var(*this, internal_tags::f_1<DataType>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  get<0>(*f_1_times_x_over_r) = get(f_1) * get<0>(x_over_r);
  get<1>(*f_1_times_x_over_r) = get(f_1) * get<1>(x_over_r);
  get<2>(*f_1_times_x_over_r) = get(f_1) * get<2>(x_over_r);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_2<DataType> /*meta*/) const {
  const auto& d_spatial_metric_rr =
      cache->get_var(*this, internal_tags::d_spatial_metric_rr<DataType>{});
  const auto& d_f_0 = cache->get_var(*this, internal_tags::d_f_0<DataType>{});
  const auto& f_1 = cache->get_var(*this, internal_tags::f_1<DataType>{});

  get(*f_2) = get(d_spatial_metric_rr) - get(d_f_0) - 2. * get(f_1);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iii<DataType, 3, Frame>*>
        f_2_times_xxx_over_r_cubed,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame> /*meta*/) const {
  const auto& f_2 = cache->get_var(*this, internal_tags::f_2<DataType>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = i; j < 3; j++) {
      for (size_t k = j; k < 3; k++) {
        f_2_times_xxx_over_r_cubed->get(i, j, k) =
            get(f_2) * x_over_r.get(i) * x_over_r.get(j) * x_over_r.get(k);
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_3,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_3<DataType> /*meta*/) const {
  const auto& one_over_r =
      cache->get_var(*this, internal_tags::one_over_r<DataType>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType>{});
  const auto& one_over_spatial_metric_rr = cache->get_var(
      *this, internal_tags::one_over_spatial_metric_rr<DataType>{});

  get(*f_3) = get(one_over_r) * get(two_m_over_m_plus_r_squared) *
              get(one_over_spatial_metric_rr);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> f_4,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::f_4<DataType> /*meta*/) const {
  const auto& f_3 = cache->get_var(*this, internal_tags::f_3<DataType>{});
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType>{});
  const auto& two_m_over_m_plus_r_cubed = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_cubed<DataType>{});
  const auto& d_spatial_metric_rr =
      cache->get_var(*this, internal_tags::d_spatial_metric_rr<DataType>{});
  const auto& one_over_spatial_metric_rr = cache->get_var(
      *this, internal_tags::one_over_spatial_metric_rr<DataType>{});

  get(*f_4) = -get(f_3) -
              get(two_m_over_m_plus_r_cubed) * get(one_over_spatial_metric_rr) /
                  solution_.mass() -
              get(d_spatial_metric_rr) * get(two_m_over_m_plus_r_squared) *
                  square(get(one_over_spatial_metric_rr));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  const auto& one_over_spatial_metric_rr = cache->get_var(
      *this, internal_tags::one_over_spatial_metric_rr<DataType>{});

  get(*lapse) = sqrt(get(one_over_spatial_metric_rr));
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        neg_half_lapse_cubed_times_d_spatial_metric_rr,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::neg_half_lapse_cubed_times_d_spatial_metric_rr<
        DataType> /*meta*/) const {
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& d_spatial_metric_rr =
      cache->get_var(*this, internal_tags::d_spatial_metric_rr<DataType>{});

  get(*neg_half_lapse_cubed_times_d_spatial_metric_rr) =
      -0.5 * cube(get(lapse)) * get(d_spatial_metric_rr);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) const {
  const auto& two_m_over_m_plus_r_squared = cache->get_var(
      *this, internal_tags::two_m_over_m_plus_r_squared<DataType>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});
  const auto& one_over_spatial_metric_rr = cache->get_var(
      *this, internal_tags::one_over_spatial_metric_rr<DataType>{});

  ::tenex::evaluate<ti::I>(shift, two_m_over_m_plus_r_squared() *
                                      x_over_r(ti::I) *
                                      one_over_spatial_metric_rr());
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    const gsl::not_null<CachedBuffer*> cache,
    DerivShift<DataType, Frame> /*meta*/) const {
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});
  const auto& f_3 = cache->get_var(*this, internal_tags::f_3<DataType>{});
  const auto& f_4 = cache->get_var(*this, internal_tags::f_4<DataType>{});

  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = k; i < 3; ++i) {
      if (i != k) {
        deriv_shift->get(k, i) = get(f_4) * x_over_r.get(i) * x_over_r.get(k);
        deriv_shift->get(i, k) = deriv_shift->get(k, i);  // symmetry
      } else {
        deriv_shift->get(k, i) = get(f_4) * square(x_over_r.get(i)) + get(f_3);
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const {
  const auto& f_0 = cache->get_var(*this, internal_tags::f_0<DataType>{});
  const auto& spatial_metric_rr_minus_f_0 = cache->get_var(
      *this, internal_tags::spatial_metric_rr_minus_f_0<DataType>{});
  const auto& x_over_r =
      cache->get_var(*this, internal_tags::x_over_r<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      spatial_metric->get(i, j) =
          get(spatial_metric_rr_minus_f_0) * x_over_r.get(i) * x_over_r.get(j);
      if (i == j) {
        spatial_metric->get(i, j) += get(f_0);
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const {
  const auto& d_f_0_times_x_over_r = cache->get_var(
      *this, internal_tags::d_f_0_times_x_over_r<DataType, Frame>{});
  const auto& f_1_times_x_over_r = cache->get_var(
      *this, internal_tags::f_1_times_x_over_r<DataType, Frame>{});
  const auto& f_2_times_xxx_over_r_cubed = cache->get_var(
      *this, internal_tags::f_2_times_xxx_over_r_cubed<DataType, Frame>{});

  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        deriv_spatial_metric->get(k, i, j) =
            f_2_times_xxx_over_r_cubed.get(k, i, j);
        if (i == j) {
          deriv_spatial_metric->get(k, i, j) += d_f_0_times_x_over_r.get(k);
        }
        if (j == k) {
          deriv_spatial_metric->get(k, i, j) += f_1_times_x_over_r.get(i);
        }
        if (i == k) {
          deriv_spatial_metric->get(k, i, j) += f_1_times_x_over_r.get(j);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) const {
  std::fill(dt_spatial_metric->begin(), dt_spatial_metric->end(), 0.);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> det_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::DetSpatialMetric<DataType> /*meta*/) const {
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3, Frame>{});

  *det_spatial_metric = determinant(spatial_metric);
}

template <typename DataType, typename Frame>
void HarmonicKerr::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> one_over_det_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::one_over_det_spatial_metric<DataType> /*meta*/) const {
  const auto& det_spatial_metric =
      cache->get_var(*this, gr::Tags::DetSpatialMetric<DataType>{});

  get(*one_over_det_spatial_metric) = 1. / get(det_spatial_metric);
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
HarmonicKerr::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    DerivLapse<DataType, Frame> /*meta*/) {
  const auto& neg_half_lapse_cubed_times_d_spatial_metric_rr = get_var(
      computer, internal_tags::neg_half_lapse_cubed_times_d_spatial_metric_rr<
                    DataType>{});
  const auto& x_over_r =
      get_var(computer, internal_tags::x_over_r<DataType, Frame>{});

  tnsr::i<DataType, 3, Frame> deriv_lapse{};
  for (size_t i = 0; i < 3; i++) {
    deriv_lapse.get(i) =
        get(neg_half_lapse_cubed_times_d_spatial_metric_rr) * x_over_r.get(i);
  }

  return deriv_lapse;
}

template <typename DataType, typename Frame>
Scalar<DataType> HarmonicKerr::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) {
  const auto& r = get(get_var(computer, internal_tags::r<DataType>{}));
  return make_with_value<Scalar<DataType>>(r, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
HarmonicKerr::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/) {
  const auto& r = get(get_var(computer, internal_tags::r<DataType>{}));
  return make_with_value<tnsr::I<DataType, 3, Frame>>(r, 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType> HarmonicKerr::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) {
  const auto& det_spatial_metric =
      get_var(computer, gr::Tags::DetSpatialMetric<DataType>{});
  return Scalar<DataType>(sqrt(get(det_spatial_metric)));
}

template <typename DataType, typename Frame>
tnsr::II<DataType, 3, Frame>
HarmonicKerr::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/) {
  const auto& spatial_metric =
      get_var(computer, gr::Tags::SpatialMetric<DataType, 3, Frame>{});
  const DataType& spatial_metric_00 = get<0, 0>(spatial_metric);
  const DataType& spatial_metric_01 = get<0, 1>(spatial_metric);
  const DataType& spatial_metric_02 = get<0, 2>(spatial_metric);
  const DataType& spatial_metric_11 = get<1, 1>(spatial_metric);
  const DataType& spatial_metric_12 = get<1, 2>(spatial_metric);
  const DataType& spatial_metric_22 = get<2, 2>(spatial_metric);
  const auto& one_over_det_spatial_metric =
      get_var(computer, internal_tags::one_over_det_spatial_metric<DataType>{});

  tnsr::II<DataType, 3, Frame> inverse_spatial_metric{};
  get<0, 0>(inverse_spatial_metric) =
      (spatial_metric_11 * spatial_metric_22 - square(spatial_metric_12)) *
      get(one_over_det_spatial_metric);
  get<0, 1>(inverse_spatial_metric) = (spatial_metric_12 * spatial_metric_02 -
                                       spatial_metric_22 * spatial_metric_01) *
                                      get(one_over_det_spatial_metric);
  get<0, 2>(inverse_spatial_metric) = (spatial_metric_01 * spatial_metric_12 -
                                       spatial_metric_02 * spatial_metric_11) *
                                      get(one_over_det_spatial_metric);
  get<1, 1>(inverse_spatial_metric) = (spatial_metric_22 * spatial_metric_00 -
                                       spatial_metric_02 * spatial_metric_02) *
                                      get(one_over_det_spatial_metric);
  get<1, 2>(inverse_spatial_metric) = (spatial_metric_02 * spatial_metric_01 -
                                       spatial_metric_00 * spatial_metric_12) *
                                      get(one_over_det_spatial_metric);
  get<2, 2>(inverse_spatial_metric) = (spatial_metric_00 * spatial_metric_11 -
                                       spatial_metric_01 * spatial_metric_01) *
                                      get(one_over_det_spatial_metric);

  return inverse_spatial_metric;
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame>
HarmonicKerr::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/) {
  return gr::extrinsic_curvature(
      get_var(computer, gr::Tags::Lapse<DataType>{}),
      get_var(computer, gr::Tags::Shift<DataType, 3, Frame>{}),
      get_var(computer, DerivShift<DataType, Frame>{}),
      get_var(computer, gr::Tags::SpatialMetric<DataType, 3, Frame>{}),
      get_var(computer,
              ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>{}),
      get_var(computer, DerivSpatialMetric<DataType, Frame>{}));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                               \
  template class HarmonicKerr::IntermediateVars<DTYPE(data), FRAME(data)>; \
  template class HarmonicKerr::IntermediateComputer<DTYPE(data), FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions
