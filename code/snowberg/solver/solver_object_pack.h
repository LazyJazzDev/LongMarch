#pragma once
#include "snowberg/solver/solver_element.h"
#include "snowberg/solver/solver_util.h"

namespace snowberg::solver {

struct ObjectPack {
  std::vector<Vector3<float>> x;
  std::vector<Vector3<float>> v;
  std::vector<float> m;

  std::vector<ElementStretching> stretchings;
  std::vector<uint32_t> stretching_indices;

  std::vector<ElementBending> bendings;
  std::vector<uint32_t> bending_indices;

  static ObjectPack CreateFromMesh(const std::vector<Vector3<float>> &positions,
                                   const std::vector<uint32_t> &indices,
                                   const Matrix3<float> &rotation = Matrix3<float>::Identity(),
                                   const Vector3<float> &translation = Vector3<float>::Zero(),
                                   float mesh_mass = 1.0f,
                                   float young = 3e3f,
                                   float poisson = 0.2f,
                                   float bending_stiffness = 0.03f,
                                   float damping = 1e-6f,
                                   float sigma_lb = -1.0f,
                                   float sigma_ub = -1.0f,
                                   float elastic_limit = 4.0f);

  static ObjectPack CreateGridCloth(const std::vector<Vector3<float>> &pos_grid,
                                    int n_row,
                                    int n_col,
                                    float total_mass = 1.0f,
                                    float young = 3e3f,
                                    float poisson = 0.2f,
                                    float bending_stiffness = 0.03f,
                                    float damping = 1e-6f,
                                    float sigma_lb = -1.0f,
                                    float sigma_ub = -1.0f,
                                    float elastic_limit = 4.0f);

 private:
  void PushStretching(int u, int v, int w, float mu, float lambda, float damping, float sigma_lb, float sigma_ub);
  void PushBending(int a, int b, int c, int d, float stiffness, float damping, float elastic_limit);
};

struct ObjectPackView {
  std::vector<int> particle_ids;
  std::vector<int> stretching_ids;
  std::vector<int> bending_ids;
};

}  // namespace snowberg::solver
