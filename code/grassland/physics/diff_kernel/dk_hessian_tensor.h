#pragma once
#include "grassland/physics/physics_util.h"

namespace grassland {

template <typename Real, int OutputDim, int InputDim>
struct HessianTensor {
  Eigen::Matrix<Real, InputDim, InputDim> m[OutputDim];

  LM_DEVICE_FUNC HessianTensor() {
    for (int i = 0; i < OutputDim; i++) {
      m[i].setZero();
    }
  }

  template <int NewOutputDim>
  LM_DEVICE_FUNC friend HessianTensor<Real, NewOutputDim, InputDim> operator*(
      const Eigen::Matrix<Real, NewOutputDim, OutputDim> &A,
      const HessianTensor<Real, OutputDim, InputDim> &H) {
    HessianTensor<Real, NewOutputDim, InputDim> new_H;
    for (int i = 0; i < NewOutputDim; i++) {
      new_H.m[i].setZero();
      for (int j = 0; j < OutputDim; j++) {
        new_H.m[i] += A(i, j) * H.m[j];
      }
    }
    return new_H;
  }

  template <int NewInputDim>
  LM_DEVICE_FUNC friend HessianTensor<Real, OutputDim, NewInputDim> operator*(
      const HessianTensor<Real, OutputDim, InputDim> &H,
      const Eigen::Matrix<Real, InputDim, NewInputDim> &A) {
    HessianTensor<Real, OutputDim, NewInputDim> new_H;
    for (int i = 0; i < OutputDim; i++) {
      new_H.m[i] = A.transpose() * H.m[i] * A;
    }
    return new_H;
  }

  LM_DEVICE_FUNC HessianTensor operator+(const HessianTensor &H) const {
    HessianTensor new_H;
    for (int i = 0; i < OutputDim; i++) {
      new_H.m[i] = m[i] + H.m[i];
    }
    return new_H;
  }

  LM_DEVICE_FUNC HessianTensor operator-(const HessianTensor &H) const {
    HessianTensor new_H;
    for (int i = 0; i < OutputDim; i++) {
      new_H.m[i] = m[i] - H.m[i];
    }
    return new_H;
  }

  LM_DEVICE_FUNC HessianTensor operator-() const {
    HessianTensor new_H;
    for (int i = 0; i < OutputDim; i++) {
      new_H.m[i] = -m[i];
    }
    return new_H;
  }

  LM_DEVICE_FUNC HessianTensor operator*(Real a) const {
    HessianTensor new_H;
    for (int i = 0; i < OutputDim; i++) {
      new_H.m[i] = a * m[i];
    }
    return new_H;
  }

  LM_DEVICE_FUNC friend HessianTensor operator*(Real a, const HessianTensor &H) {
    return H * a;
  }

  LM_DEVICE_FUNC HessianTensor operator/(Real a) const {
    HessianTensor new_H;
    for (int i = 0; i < OutputDim; i++) {
      new_H.m[i] = m[i] / a;
    }
    return new_H;
  }

  LM_DEVICE_FUNC Real &operator()(int i, int j, int k) {
    return m[i](j, k);
  }

  LM_DEVICE_FUNC const Real &operator()(int i, int j, int k) const {
    return m[i](j, k);
  }

  friend std::ostream &operator<<(std::ostream &os, const HessianTensor &H) {
    for (int i = 0; i < OutputDim; i++) {
      os << H.m[i] << std::endl;
    }
    return os;
  }
};

}  // namespace grassland
