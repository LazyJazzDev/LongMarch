#include "snowberg/solver/solver_object_pack.h"

namespace snowberg::solver {

ObjectPack ObjectPack::CreateFromMesh(const std::vector<Vector3<float>> &positions,
                                      const std::vector<uint32_t> &indices,
                                      const Matrix3<float> &rotation,
                                      const Vector3<float> &translation,
                                      float mesh_mass,
                                      float young,
                                      float poisson,
                                      float bending_stiffness,
                                      float damping,
                                      float sigma_lb,
                                      float sigma_ub,
                                      float elastic_limit) {
  ObjectPack object_pack;
  int num_particles = positions.size();
  float particle_mass = mesh_mass / num_particles;

  for (size_t i = 0; i < num_particles; i++) {
    Vector3<float> x = rotation * positions[i] + translation;
    object_pack.x.push_back(x);
    object_pack.v.push_back(Vector3<float>::Zero());
    object_pack.m.push_back(particle_mass);
  }

  float mu = young / (2 * (1 + poisson));
  float lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));

  for (int i = 0; i < indices.size() / 3; i++) {
    object_pack.PushStretching(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2], mu, lambda, damping, sigma_lb,
                               sigma_ub);
  }

  if (bending_stiffness > 0.0) {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_map;

    for (int i = 0; i < indices.size() / 3; i++) {
      uint32_t u = indices[3 * i];
      uint32_t v = indices[3 * i + 1];
      uint32_t w = indices[3 * i + 2];
      edge_map[{u, v}] = w;
      edge_map[{v, w}] = u;
      edge_map[{w, u}] = v;
    }

    for (auto &[u, v] : edge_map) {
      auto [b, c] = u;
      if (b > c || edge_map.find({c, b}) == edge_map.end()) {
        continue;
      }
      auto &a = edge_map[{c, b}];
      auto &d = v;
      object_pack.PushBending(a, b, c, d, bending_stiffness, damping, elastic_limit);
    }
  }
  return object_pack;
}

ObjectPack ObjectPack::CreateGridCloth(const std::vector<Vector3<float>> &pos_grid,
                                       int n_row,
                                       int n_col,
                                       float total_mass,
                                       float young,
                                       float poisson,
                                       float bending_stiffness,
                                       float damping,
                                       float sigma_lb,
                                       float sigma_ub,
                                       float elastic_limit) {
  ObjectPack object_pack;
  int num_particle = n_row * n_col;
  object_pack.x.resize(num_particle);
  object_pack.v.resize(num_particle);
  object_pack.m.resize(num_particle);
  float particle_mass = total_mass / num_particle;
  for (int i = 0; i < num_particle; i++) {
    object_pack.x[i] = pos_grid[i];
    object_pack.v[i] = Vector3<float>::Zero();
    object_pack.m[i] = particle_mass;
  }

  float mu = young / (2 * (1 + poisson));
  float lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));

  for (int i = 0; i < n_col - 1; i++) {
    int i1 = i + 1;
    for (int j = 0; j < n_row - 1; j++) {
      int j1 = j + 1;
      int v00 = i * n_row + j;
      int v01 = i * n_row + j1;
      int v10 = i1 * n_row + j;
      int v11 = i1 * n_row + j1;
      object_pack.PushStretching(v00, v01, v10, mu * .25, lambda * .25, damping, sigma_lb, sigma_ub);
      object_pack.PushStretching(v11, v01, v10, mu * .25, lambda * .25, damping, sigma_lb, sigma_ub);
      object_pack.PushStretching(v00, v01, v11, mu * .25, lambda * .25, damping, sigma_lb, sigma_ub);
      object_pack.PushStretching(v00, v10, v11, mu * .25, lambda * .25, damping, sigma_lb, sigma_ub);
      object_pack.PushBending(v01, v00, v11, v10, bending_stiffness * .5, damping, elastic_limit);
      object_pack.PushBending(v00, v01, v10, v11, bending_stiffness * .5, damping, elastic_limit);
    }
  }

  for (int i = 1; i < n_col - 1; i++) {
    int i_1 = i - 1;
    int i1 = i + 1;
    for (int j = 0; j < n_row - 1; j++) {
      int j1 = j + 1;
      int v00 = i * n_row + j;
      int v01 = i * n_row + j1;
      int v10 = i1 * n_row + j;
      int v11 = i1 * n_row + j1;
      int v_10 = i_1 * n_row + j;
      int v_11 = i_1 * n_row + j1;

      object_pack.PushBending(v_10, v00, v01, v10, bending_stiffness * .25, damping, elastic_limit);
      object_pack.PushBending(v_11, v00, v01, v10, bending_stiffness * .25, damping, elastic_limit);
      object_pack.PushBending(v_10, v00, v01, v11, bending_stiffness * .25, damping, elastic_limit);
      object_pack.PushBending(v_11, v00, v01, v11, bending_stiffness * .25, damping, elastic_limit);
    }
  }

  for (int i = 0; i < n_col - 1; i++) {
    int i1 = i + 1;
    for (int j = 1; j < n_row - 1; j++) {
      int j_1 = j - 1;
      int j1 = j + 1;
      int v00 = i * n_row + j;
      int v01 = i * n_row + j1;
      int v0_1 = i * n_row + j_1;
      int v10 = i1 * n_row + j;
      int v11 = i1 * n_row + j1;
      int v1_1 = i1 * n_row + j_1;

      object_pack.PushBending(v0_1, v00, v10, v01, bending_stiffness * .25, damping, elastic_limit);
      object_pack.PushBending(v1_1, v00, v10, v01, bending_stiffness * .25, damping, elastic_limit);
      object_pack.PushBending(v0_1, v00, v10, v11, bending_stiffness * .25, damping, elastic_limit);
      object_pack.PushBending(v1_1, v00, v10, v11, bending_stiffness * .25, damping, elastic_limit);
    }
  }

  return object_pack;
}

void ObjectPack::PushStretching(int u,
                                int v,
                                int w,
                                float mu,
                                float lambda,
                                float damping,
                                float sigma_lb,
                                float sigma_ub) {
  ElementStretching stretching;
  stretching.mu = mu;
  stretching.lambda = lambda;

  const Vector3<float> x0 = x[u];
  const Vector3<float> x1 = x[v];
  const Vector3<float> x2 = x[w];

  Matrix2<float> s;
  s(0, 0) = (x1 - x0).norm();
  s(1, 0) = 0.0;
  s(0, 1) = (x2 - x0).dot(x1 - x0) / s(0, 0);
  s(1, 1) = std::sqrt((x2 - x0).squaredNorm() - s(0, 1) * s(0, 1));

  stretching.Dm = s;
  stretching.area = 0.5f * stretching.Dm.determinant();
  stretching.sigma_lb = sigma_lb;
  stretching.sigma_ub = sigma_ub;
  stretching.damping = damping;

  stretchings.push_back(stretching);
  stretching_indices.push_back(u);
  stretching_indices.push_back(v);
  stretching_indices.push_back(w);
}

void ObjectPack::PushBending(int a, int b, int c, int d, float stiffness, float damping, float elastic_limit) {
  ElementBending bending;
  DihedralAngle<float> dihedral_angle;
  Matrix<float, 3, 4> V;
  V << x[a], x[b], x[c], x[d];

  const Vector3<float> e = V.col(1) - V.col(2);

  const float h = (e.cross(V.col(0) - V.col(1)).norm() + e.cross(V.col(2) - V.col(3)).norm()) / e.norm();

  bending.stiffness = stiffness * e.norm() * 6.0f / h;
  bending.damping = damping;
  bending.theta_rest = dihedral_angle(V).value();
  bending.elastic_limit = elastic_limit;

  bendings.push_back(bending);
  bending_indices.push_back(a);
  bending_indices.push_back(b);
  bending_indices.push_back(c);
  bending_indices.push_back(d);
}

}  // namespace snowberg::solver
