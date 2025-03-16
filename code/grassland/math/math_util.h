#pragma once
#include "Eigen/Eigen"
#include "glm/glm.hpp"
#include "grassland/util/util.h"

namespace grassland {
using Eigen::Matrix;
using Eigen::Matrix2;
using Eigen::Matrix3;
using Eigen::Matrix4;
using Eigen::RowVector2;
using Eigen::RowVector3;
using Eigen::RowVector4;
using Eigen::Vector;
using Eigen::Vector2;
using Eigen::Vector3;
using Eigen::Vector4;

template <typename Scalar>
LM_DEVICE_FUNC Scalar Eps();

template <>
LM_DEVICE_FUNC inline float Eps<float>() {
  return 1e-4f;
}

template <>
LM_DEVICE_FUNC inline double Eps<double>() {
  return 1e-8;
}

template <typename Scalar>
LM_DEVICE_FUNC int Sign(Scalar x) {
  if (x < 0) {
    return -1;
  } else if (x > 0) {
    return 1;
  } else {
    return 0;
  }
}

template <typename Scalar>
LM_DEVICE_FUNC Scalar PI() {
  return 3.14159265358979323846264338327950288419716939937510;
}

template <typename Scalar, int row, int col>
LM_DEVICE_FUNC glm::mat<col, row, Scalar> EigenToGLM(const Matrix<Scalar, row, col> &m) {
  glm::mat<col, row, Scalar> result;
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      result[i][j] = m(j, i);
    }
  }
  return result;
}

template <typename Scalar, int dim>
LM_DEVICE_FUNC glm::vec<dim, Scalar> EigenToGLM(const Vector<Scalar, dim> &v) {
  glm::vec<dim, Scalar> result;
  for (int i = 0; i < dim; i++) {
    result[i] = v[i];
  }
  return result;
}

template <typename Scalar, int row, int col>
LM_DEVICE_FUNC Matrix<Scalar, row, col> GLMToEigen(const glm::mat<col, row, Scalar> &m) {
  Matrix<Scalar, row, col> result;
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      result(j, i) = m[i][j];
    }
  }
  return result;
}

template <typename Scalar, int dim>
LM_DEVICE_FUNC Vector<Scalar, dim> EigenToGLM(const glm::vec<dim, Scalar> &v) {
  Vector<Scalar, dim> result;
  for (int i = 0; i < dim; i++) {
    result[i] = v[i];
  }
  return result;
}

}  // namespace grassland
