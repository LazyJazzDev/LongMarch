#pragma once

#include "grassland/data_structure/data_structure.h"
#include "grassland/geometry/geometry_util.h"

namespace grassland::geometry {

using offset_t = data_structure::offset_t;

template <typename ContentType,
          typename Scalar = float,
          typename GridType = data_structure::LinearGrid<ContentType>>
class Field {
 public:
  static Matrix<Scalar, 3, 4> CreateTransform(Scalar delta_x,
                                              const Vector3<Scalar> &offset) {
    Matrix<Scalar, 3, 4> transform = Matrix<Scalar, 3, 4>::Zero();
    transform(0, 0) = delta_x;
    transform(1, 1) = delta_x;
    transform(2, 2) = delta_x;
    transform.block(0, 3, 3, 1) = offset;
    return transform;
  }

  static Matrix<Scalar, 3, 4> CreateTransform(const Vector3<Scalar> &delta,
                                              const Vector3<Scalar> &offset) {
    Matrix<Scalar, 3, 4> transform = Matrix<Scalar, 3, 4>::Zero();
    transform(0, 0) = delta[0];
    transform(1, 1) = delta[1];
    transform(2, 2) = delta[2];
    transform.block(0, 3, 3, 1) = offset;
    return transform;
  }

  Field(size_t width,
        size_t height,
        size_t depth,
        const Matrix<Scalar, 3, 4> &transform,
        const ContentType &default_value = ContentType{})
      : grid_(width, height, depth, default_value) {
    set_transform(transform);
  }

  Field(size_t width,
        size_t height,
        size_t depth,
        Scalar delta_x,
        const Vector3<Scalar> &offset,
        const ContentType &default_value = ContentType{})
      : Field(width,
              height,
              depth,
              CreateTransform(delta_x, offset),
              default_value) {
  }

  Field(size_t width,
        size_t height,
        size_t depth,
        const Vector3<Scalar> &delta,
        const Vector3<Scalar> &offset,
        const ContentType &default_value = ContentType{})
      : Field(width,
              height,
              depth,
              CreateTransform(delta, offset),
              default_value) {
  }

  Field(const Matrix<Scalar, 3, 4> &transform, const GridType &grid)
      : grid_(grid) {
    set_transform(transform);
  }

  Field(Scalar delta_x, const Vector3<Scalar> &offset, const GridType &grid)
      : grid_(grid) {
    set_transform(CreateTransform(delta_x, offset));
  }

  Field(const Vector3<Scalar> &delta,
        const Vector3<Scalar> &offset,
        const GridType &grid)
      : grid_(grid) {
    set_transform(CreateTransform(delta, offset));
  }

  ContentType &operator()(offset_t x, offset_t y, offset_t z) {
    return grid_(x, y, z);
  }

  const ContentType &operator()(offset_t x, offset_t y, offset_t z) const {
    return grid_(x, y, z);
  }

  Vector3<Scalar> get_position(offset_t x, offset_t y, offset_t z) const {
    return transform_ * Vector4<Scalar>(x, y, z, 1);
  }

  Vector3<Scalar> to_world_position(const Vector3<Scalar> &pos) const {
    Vector3<Scalar> pos_transformed = transform_ * pos.homogeneous();
    return pos_transformed;
  }

  Vector3<Scalar> to_grid_position(const Vector3<Scalar> &pos) const {
    Vector3<Scalar> pos_transformed = inv_transform_ * pos.homogeneous();
    return pos_transformed;
  }

  ContentType operator()(const Vector3<Scalar> &pos) const {
    Vector3<Scalar> pos_transformed = inv_transform_ * pos.homogeneous();
    return grid_.sample(pos_transformed[0], pos_transformed[1],
                        pos_transformed[2]);
  }

  size_t width() const {
    return grid_.width();
  }

  size_t height() const {
    return grid_.height();
  }

  size_t depth() const {
    return grid_.depth();
  }

  GridType &grid() {
    return grid_;
  }

  const GridType &grid() const {
    return grid_;
  }

  Matrix<Scalar, 3, 4> get_transform() const {
    return transform_;
  }

  Matrix<Scalar, 3, 4> get_inv_transform() const {
    return inv_transform_;
  }

  void set_transform(const Matrix<Scalar, 3, 4> &transform) {
    transform_ = transform;
    Matrix4<Scalar> transform4;
    transform4.block(0, 0, 3, 4) = transform_;
    transform4.row(3) = Matrix<Scalar, 1, 4>::Zero();
    transform4(3, 3) = 1;
    transform4 = transform4.inverse().eval();
    inv_transform_ = transform4.block(0, 0, 3, 4);
  }

 private:
  GridType grid_;
  Matrix<Scalar, 3, 4> transform_;
  Matrix<Scalar, 3, 4> inv_transform_;
};
}  // namespace grassland::geometry
