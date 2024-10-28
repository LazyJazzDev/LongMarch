#pragma once
#include "grassland/data_structure/grid/grid_util.h"

namespace grassland::data_structure {
template <typename ContentType>
class LinearGridView {
 public:
  LinearGridView(size_t width, size_t height, ContentType *data)
      : LinearGridView(width, height, 1, data) {
  }

  LinearGridView(size_t width, size_t height, size_t depth, ContentType *data)
      : width_(width),
        height_(height),
        depth_(depth),
        data_(data),
        x_stride_(1),
        y_stride_(width),
        z_stride_(width * height) {
  }

  LM_DEVICE_FUNC ContentType &operator[](offset_t offset) {
    return data_[offset];
  }

  LM_DEVICE_FUNC const ContentType &operator[](offset_t offset) const {
    return data_[offset];
  }

  LM_DEVICE_FUNC offset_t offset(offset_t x, offset_t y) const {
    return x * x_stride_ + y * y_stride_;
  }

  LM_DEVICE_FUNC offset_t offset(offset_t x, offset_t y, offset_t z) const {
    return x * x_stride_ + y * y_stride_ + z * z_stride_;
  }

  LM_DEVICE_FUNC ContentType &operator()(offset_t x, offset_t y) {
    return data_[x * x_stride_ + y * y_stride_];
  }

  LM_DEVICE_FUNC const ContentType &operator()(offset_t x, offset_t y) const {
    return data_[x * x_stride_ + y * y_stride_];
  }

  LM_DEVICE_FUNC ContentType &operator()(offset_t x, offset_t y, offset_t z) {
    return data_[x * x_stride_ + y * y_stride_ + z * z_stride_];
  }

  LM_DEVICE_FUNC const ContentType &operator()(offset_t x,
                                               offset_t y,
                                               offset_t z) const {
    return data_[x * x_stride_ + y * y_stride_ + z * z_stride_];
  }

  LM_DEVICE_FUNC size_t width() const {
    return width_;
  }

  LM_DEVICE_FUNC size_t height() const {
    return height_;
  }

  LM_DEVICE_FUNC size_t depth() const {
    return depth_;
  }

  LM_DEVICE_FUNC ContentType get(offset_t x, offset_t y, offset_t z) const {
    return data_[offset(x, y, z)];
  }

  LM_DEVICE_FUNC ContentType get_clamped(offset_t x,
                                         offset_t y,
                                         offset_t z) const {
    return data_[offset(std::clamp(x, offset_t(0), offset_t(width_ - 1)),
                        std::clamp(y, offset_t(0), offset_t(height_ - 1)),
                        std::clamp(z, offset_t(0), offset_t(depth_ - 1)))];
  }

  template <class Scalar>
  LM_DEVICE_FUNC ContentType sample(Scalar x, Scalar y, Scalar z) const {
    offset_t x0 = static_cast<offset_t>(std::floor(x));
    offset_t y0 = static_cast<offset_t>(std::floor(y));
    offset_t z0 = static_cast<offset_t>(std::floor(z));
    x -= x0;
    y -= y0;
    z -= z0;
    return get_clamped(x0, y0, z0) * ((1 - x) * (1 - y) * (1 - z)) +
           get_clamped(x0 + 1, y0, z0) * (x * (1 - y) * (1 - z)) +
           get_clamped(x0, y0 + 1, z0) * ((1 - x) * y * (1 - z)) +
           get_clamped(x0 + 1, y0 + 1, z0) * (x * y * (1 - z)) +
           get_clamped(x0, y0, z0 + 1) * ((1 - x) * (1 - y) * z) +
           get_clamped(x0 + 1, y0, z0 + 1) * (x * (1 - y) * z) +
           get_clamped(x0, y0 + 1, z0 + 1) * ((1 - x) * y * z) +
           get_clamped(x0 + 1, y0 + 1, z0 + 1) * (x * y * z);
  }

  LM_DEVICE_FUNC ContentType *data() {
    return data_;
  }

  LM_DEVICE_FUNC const ContentType *data() const {
    return data_;
  }

  LM_DEVICE_FUNC size_t x_stride() const {
    return x_stride_;
  }

  LM_DEVICE_FUNC size_t y_stride() const {
    return y_stride_;
  }

  LM_DEVICE_FUNC size_t z_stride() const {
    return z_stride_;
  }

 private:
  ContentType *data_;
  size_t width_;
  size_t height_;
  size_t depth_;
  size_t x_stride_;
  size_t y_stride_;
  size_t z_stride_;
};
}  // namespace grassland::data_structure
