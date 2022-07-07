// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/MomentumDensity.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {
template <size_t SpatialDim>
void momentum_density(

    gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*> result,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) {
  result = (pi * phi);
}

template <size_t SpatialDim>
tnsr::i<DataVector, SpatialDim, Frame::Inertial> momentum_density(
    const Scalar<DataVector>* pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) {
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> result{get(phi).size()};
  momentum_density(make_not_null(&result), pi, phi);
  return result;
}

}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)
template void ScalarWave::momentum_density(
    gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*> result,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
