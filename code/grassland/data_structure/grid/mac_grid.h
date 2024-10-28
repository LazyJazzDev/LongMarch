#pragma once
#include "grassland/data_structure/data_structure_util.h"
#include "grassland/data_structure/grid/linear_grid.h"

namespace grassland::data_structure {
template <typename ContentType, typename BaseGridType = LinearGrid<ContentType>>
class MACGrid {
 public:
  MACGrid(size_t width,
          size_t height,
          size_t depth,
          const ContentType &default_value = ContentType{})
      : MACGrid(width,
                height,
                depth,
                default_value,
                default_value,
                default_value) {
  }

  MACGrid(size_t width,
          size_t height,
          size_t depth,
          const ContentType &default_value_u,
          const ContentType &default_value_v,
          const ContentType &default_value_w)
      : u_(width + 1, height, depth, default_value_u),
        v_(width, height + 1, depth, default_value_v),
        w_(width, height, depth + 1, default_value_w) {
  }

  template <class OtherBaseGridType>
  MACGrid(MACGrid<ContentType, OtherBaseGridType> &other) {
    u_ = other.u();
    v_ = other.v();
    w_ = other.w();
  }

  BaseGridType &u() {
    return u_;
  }

  const BaseGridType &u() const {
    return u_;
  }

  BaseGridType &v() {
    return v_;
  }

  const BaseGridType &v() const {
    return v_;
  }

  BaseGridType &w() {
    return w_;
  }

  const BaseGridType &w() const {
    return w_;
  }

  template <class Scalar>
  ContentType sample_u(Scalar x, Scalar y, Scalar z) const {
    return u_.sample(x, y - 0.5, z - 0.5);
  }

  template <class Scalar>
  ContentType sample_v(Scalar x, Scalar y, Scalar z) const {
    return v_.sample(x - 0.5, y, z - 0.5);
  }

  template <class Scalar>
  ContentType sample_w(Scalar x, Scalar y, Scalar z) const {
    return w_.sample(x - 0.5, y - 0.5, z);
  }

  size_t width() const {
    return u_.width() - 1;
  }

  size_t height() const {
    return v_.height() - 1;
  }

  size_t depth() const {
    return w_.depth() - 1;
  }

 private:
  BaseGridType u_;
  BaseGridType v_;
  BaseGridType w_;
};
}  // namespace grassland::data_structure
