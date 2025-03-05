#pragma once
#include "Eigen/Eigen"
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

struct VertexBufferView {
  const void *data;
  size_t stride;
  VertexBufferView() : data(nullptr), stride(0) {
  }
  VertexBufferView(const void *data_, size_t stride_, size_t offset_ = 0)
      : data(static_cast<const char *>(data_) + offset_), stride(stride_) {
  }

  template <typename T>
  VertexBufferView(const T *data_, size_t offset = 0) : data(data_), stride(sizeof(T)) {
    data = static_cast<const char *>(data) + offset;
  }

  template <typename T>
  LM_DEVICE_FUNC const T &Get(size_t index, size_t offset = 0) const {
    return *reinterpret_cast<const T *>(static_cast<const char *>(data) + stride * index + offset);
  }

  template <typename T>
  LM_DEVICE_FUNC T &At(size_t index, size_t offset = 0) {
    return *reinterpret_cast<T *>(static_cast<char *>(const_cast<void *>(data)) + stride * index + offset);
  }
};

}  // namespace grassland
