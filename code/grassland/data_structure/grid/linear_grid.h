#pragma once

#include "grassland/data_structure/grid/grid_util.h"
#include "grassland/data_structure/grid/linear_grid_view.h"

namespace grassland::data_structure {
template <typename ContentType>
class LinearGrid {
 public:
  LinearGrid(size_t width,
             size_t height,
             const ContentType &default_value = ContentType{})
      : LinearGrid(width, height, 1, default_value) {
  }

  LinearGrid(size_t width,
             size_t height,
             size_t depth,
             const ContentType &default_value = ContentType{})
      : width_(width),
        height_(height),
        depth_(depth),
        buffer_(width * height * depth, default_value),
        x_stride_(1),
        y_stride_(width),
        z_stride_(width * height) {
  }

  ~LinearGrid() = default;

  ContentType &operator[](offset_t offset) {
    return buffer_[offset];
  }

  const ContentType &operator[](offset_t offset) const {
    return buffer_[offset];
  }

  offset_t offset(offset_t x, offset_t y) const {
    return x * x_stride_ + y * y_stride_;
  }

  offset_t offset(offset_t x, offset_t y, offset_t z) const {
    return x * x_stride_ + y * y_stride_ + z * z_stride_;
  }

  ContentType &operator()(offset_t x, offset_t y) {
    return buffer_[x * x_stride_ + y * y_stride_];
  }

  const ContentType &operator()(offset_t x, offset_t y) const {
    return buffer_[x * x_stride_ + y * y_stride_];
  }

  ContentType &operator()(offset_t x, offset_t y, offset_t z) {
    return buffer_[x * x_stride_ + y * y_stride_ + z * z_stride_];
  }

  const ContentType &operator()(offset_t x, offset_t y, offset_t z) const {
    return buffer_[x * x_stride_ + y * y_stride_ + z * z_stride_];
  }

  size_t width() const {
    return width_;
  }

  size_t height() const {
    return height_;
  }

  size_t depth() const {
    return depth_;
  }

  ContentType get(offset_t x, offset_t y, offset_t z) const {
    return buffer_[offset(x, y, z)];
  }

  ContentType get_clamped(offset_t x, offset_t y, offset_t z) const {
    return buffer_[offset(std::clamp(x, offset_t(0), offset_t(width_ - 1)),
                          std::clamp(y, offset_t(0), offset_t(height_ - 1)),
                          std::clamp(z, offset_t(0), offset_t(depth_ - 1)))];
  }

  template <class Scalar>
  ContentType sample(Scalar x, Scalar y, Scalar z) const {
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

  ContentType *data() {
    return buffer_.data();
  }

  const ContentType *data() const {
    return buffer_.data();
  }

  std::vector<ContentType> &buffer() {
    return buffer_;
  }

  const std::vector<ContentType> &buffer() const {
    return buffer_;
  }

  LinearGridView<ContentType> view() {
    return {width_, height_, depth_, buffer_.data()};
  }

  operator LinearGridView<ContentType>() {
    return view();
  }

 private:
  std::vector<ContentType> buffer_;
  size_t width_;
  size_t height_;
  size_t depth_;
  size_t x_stride_;
  size_t y_stride_;
  size_t z_stride_;
};
}  // namespace grassland::data_structure
