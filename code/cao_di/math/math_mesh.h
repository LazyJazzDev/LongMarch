#pragma once

#include <filesystem>

#include "cao_di/math/math_util.h"
#include "fstream"
#include "mikktspace.h"
#include "tiny_obj_loader.h"

namespace CD {
template <typename Scalar = float>
class Mesh {
 public:
  Mesh(size_t num_vertices = 0,
       size_t num_indices = 0,
       const uint32_t *indices = nullptr,
       const Vector3<Scalar> *positions = nullptr,
       const Vector3<Scalar> *normals = nullptr,
       const Vector2<Scalar> *tex_coords = nullptr,
       const Vector3<Scalar> *tangents = nullptr);

  size_t NumVertices() const {
    return num_vertices_;
  }

  size_t NumIndices() const {
    return num_indices_;
  }

  Vector3<Scalar> *Positions() {
    return positions_.data();
  }

  Vector3<Scalar> *Normals() {
    if (normals_.empty())
      return nullptr;
    return normals_.data();
  }

  Vector3<Scalar> *Tangents() {
    if (tangents_.empty())
      return nullptr;
    return tangents_.data();
  }

  Vector2<Scalar> *TexCoords() {
    if (tex_coords_.empty())
      return nullptr;
    return tex_coords_.data();
  }

  float *Signals() {
    if (signals_.empty())
      return nullptr;
    return signals_.data();
  }

  uint32_t *Indices() {
    return indices_.data();
  }

  const Vector3<Scalar> *Positions() const {
    return positions_.data();
  }

  const Vector3<Scalar> *Normals() const {
    if (normals_.empty())
      return nullptr;
    return normals_.data();
  }

  const Vector3<Scalar> *Tangents() const {
    if (tangents_.empty())
      return nullptr;
    return tangents_.data();
  }

  const Vector2<Scalar> *TexCoords() const {
    if (tex_coords_.empty())
      return nullptr;
    return tex_coords_.data();
  }

  const float *Signals() const {
    if (signals_.empty())
      return nullptr;
    return signals_.data();
  }

  const uint32_t *Indices() const {
    return indices_.data();
  }

  int LoadObjFile(const std::string &filename);

  int SaveObjFile(const std::string &filename) const;

  int SplitVertices();

  int MergeVertices();

  int GenerateNormals(Scalar merging_threshold = 0.8f);  // if all the face normals on a vertex's pairwise dot product
  // larger than merging_threshold, then merge them

  int InitializeTexCoords(const Vector2<Scalar> &tex_coord = Vector2<Scalar>{0.5, 0.5});

  int GenerateTangents();

  static Mesh<Scalar> Sphere(int precision_lon = 10, int precision_lat = -1);

 private:
  std::vector<Vector3<Scalar>> positions_;
  std::vector<Vector3<Scalar>> normals_;
  std::vector<Vector3<Scalar>> tangents_;
  std::vector<Vector2<Scalar>> tex_coords_;
  std::vector<float> signals_;
  std::vector<uint32_t> indices_;
  size_t num_vertices_{0};
  size_t num_indices_{0};
};

}  // namespace CD
