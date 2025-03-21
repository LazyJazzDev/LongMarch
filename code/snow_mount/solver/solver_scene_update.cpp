#if defined(__CUDACC__)
#include "snow_mount/solver/solver_scene.h"

#define DISPATCH_SIZE(thread_count, block_size) ((thread_count + block_size - 1) / block_size), block_size
#define DEFAULT_DISPATCH_SIZE(num_particle) DISPATCH_SIZE(num_particle, 256)

namespace snow_mount::solver {

__global__ void InitializeSolver(SceneRef scene_ref, Vector3<float> gravity, float dt) {
  int pid = threadIdx.x + blockIdx.x * blockDim.x;
  if (pid < scene_ref.num_particle) {
    scene_ref.v[pid] += gravity * dt;
    scene_ref.x[pid] = scene_ref.x_prev[pid] + scene_ref.v[pid] * dt;
  }
}

__global__ void SolveVBDParticlePosition(SceneRef scene_ref, const int *particle_indices, int num_particle, float dt) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_particle) {
    int pid = particle_indices[tid];
    int pidx = scene_ref.ParticleIndex(pid);
    Vector3<float> x = scene_ref.x[pidx];
    Vector3<float> x_prev = scene_ref.x_prev[pidx];
    Vector3<float> v = scene_ref.v[pidx];
    Vector3<float> y = x_prev + v * dt;
    float m = scene_ref.m[pidx];
    Vector3<float> f = -(m / (dt * dt)) * (x - y);
    Matrix3<float> H = (m / (dt * dt)) * Matrix3<float>::Identity();

    for (int i = 0; i < scene_ref.stretching_directory.count[pid]; i++) {
      int stretching_id = scene_ref.stretching_directory.positions[scene_ref.stretching_directory.first[pid] + i] / 3;
      auto stretching = scene_ref.stretchings[stretching_id];

      grassland::ElasticNeoHookeanSimpleTriangle<float> neo_hookean{stretching.mu, stretching.lambda, stretching.Dm};
      uint32_t u = scene_ref.stretching_indices[stretching_id * 3 + 0];
      uint32_t v = scene_ref.stretching_indices[stretching_id * 3 + 1];
      uint32_t w = scene_ref.stretching_indices[stretching_id * 3 + 2];
      int self_index = pid == u ? 0 : (pid == v ? 1 : 2);

      Matrix3<float> X;
      X << scene_ref.x[u], scene_ref.x[v], scene_ref.x[w];
      Vector3<float> jacobian = neo_hookean.Jacobian(X).block(0, 3 * self_index, 1, 3).transpose() * stretching.area;
      Matrix3<float> hessian =
          neo_hookean.Hessian(X).m[0].block(3 * self_index, 3 * self_index, 3, 3) * stretching.area;

      float k_damping = stretching.damping / dt;
      f -= jacobian + hessian * (x - x_prev) * k_damping;
      H += hessian * (1.0 + k_damping);
    }

    for (int i = 0; i < scene_ref.bending_directory.count[pid]; i++) {
      int bending_id = scene_ref.bending_directory.positions[scene_ref.bending_directory.first[pid] + i] / 4;
      auto bending = scene_ref.bendings[bending_id];

      grassland::DihedralAngle<float> dihedral_angle;
      uint32_t u = scene_ref.bending_indices[bending_id * 4 + 0];
      uint32_t v = scene_ref.bending_indices[bending_id * 4 + 1];
      uint32_t w = scene_ref.bending_indices[bending_id * 4 + 2];
      uint32_t z = scene_ref.bending_indices[bending_id * 4 + 3];
      int self_index = pid == u ? 0 : (pid == v ? 1 : (pid == w ? 2 : 3));

      Matrix<float, 3, 4> X;
      X << scene_ref.x[u], scene_ref.x[v], scene_ref.x[w], scene_ref.x[z];
      float theta = dihedral_angle(X).value() - bending.theta_rest;
      Vector3<float> jacobian = dihedral_angle.Jacobian(X).block(0, 3 * self_index, 1, 3).transpose();
      Matrix3<float> hessian = Matrix3<float>::Zero();
      hessian = dihedral_angle.SubHessian(X, self_index);
      hessian = (2.0 * theta * hessian + 2.0 * jacobian * jacobian.transpose()).derived();
      jacobian *= 2.0 * theta;

      // if (fabs(theta) > 0.0f) {
      //   printf("theta: %f jacobian: %f %f %f, current_angle: %f, theta_rest: %f\n", theta, jacobian[0], jacobian[1],
      //   jacobian[2], dihedral_angle(X).value(), bending.theta_rest);
      // }

      float k_damping = bending.damping / dt;
      f -= jacobian * bending.stiffness + hessian * (x - x_prev) * k_damping * bending.stiffness;
      H += hessian * (1.0 + k_damping) * bending.stiffness;
    }

    const float k_friction = 5.0f;
    const Vector3<float> rel_vel = (x - x_prev) / dt;

    constexpr float K_DAMPING = 1e-6;
    for (int i = 0; i < scene_ref.num_rigid_object; i++) {
      float sdf;
      Vector3<float> jacobian;
      Matrix3<float> hessian;
      RigidObjectRef rigid_object = scene_ref.rigid_objects[i];
      rigid_object.mesh_sdf.SDF(x, rigid_object.state.R, rigid_object.state.t, &sdf, &jacobian, &hessian);
      Vector3<float> r = x - sdf * jacobian - rigid_object.state.t;
      sdf -= 0.018;
      if (sdf < 0.0) {
        float k_stiffness = rigid_object.stiffness * m;
        float force_mag = -2.0 * k_stiffness * sdf;
        Matrix3<float> partial_H =
            2.0 * k_stiffness * jacobian * jacobian.transpose() + 2.0 * k_stiffness * sdf * hessian;
        f -= 2.0 * k_stiffness * sdf * jacobian + partial_H * (x - x_prev) * K_DAMPING;
        H += partial_H + partial_H * K_DAMPING;

        Vector3<float> velocity_component = rel_vel - rigid_object.state.v - rigid_object.state.omega.cross(r);
        velocity_component = velocity_component - jacobian * jacobian.transpose() * velocity_component;
        float max_friction_force = force_mag * k_friction;
        float vel_comp_norm = velocity_component.norm();
        if (vel_comp_norm > max_friction_force * dt / m) {
          f -= velocity_component / vel_comp_norm * max_friction_force;
        } else if (vel_comp_norm > 1e-9) {
          f -= velocity_component / dt * m;
        }
      }
    }
    Vector3<float> delta_x = H.inverse() * f;
    scene_ref.x[pidx] += delta_x;
  }
}

__global__ void UpdateStretchingPlasticity(SceneRef scene_ref) {
  int sid = threadIdx.x + blockIdx.x * blockDim.x;
  if (sid < scene_ref.num_stretching) {
    ElementStretching stretching = scene_ref.stretchings[sid];
    if (stretching.sigma_lb > 0.0 || stretching.sigma_ub > 0.0) {
      uint32_t u = scene_ref.stretching_indices[sid * 3 + 0];
      uint32_t v = scene_ref.stretching_indices[sid * 3 + 1];
      uint32_t w = scene_ref.stretching_indices[sid * 3 + 2];
      Matrix3<float> X;
      X << scene_ref.x[u], scene_ref.x[v], scene_ref.x[w];
      Matrix<float, 3, 2> F;
      F << X.col(1) - X.col(0), X.col(2) - X.col(0);
      Matrix<float, 3, 2> Fe = F * stretching.Dm.inverse();
      Matrix<float, 3, 2> U;
      Matrix<float, 2, 2> S;
      Matrix<float, 2, 2> Vt;
      SVD(Fe, U, S, Vt);
      if (stretching.sigma_lb > 0.0) {
        // if (S(0, 0) < stretching.theta_lb || S(1, 1) < stretching.theta_lb) {
        //   // printf("Stretching clamped (lower)\n");
        // }
        S(0, 0) = max(S(0, 0), stretching.sigma_lb);
        S(1, 1) = max(S(1, 1), stretching.sigma_lb);
      }
      if (stretching.sigma_ub > 0.0) {
        // if (S(0, 0) > stretching.theta_ub || S(1, 1) > stretching.theta_ub) {
        //   printf("Stretching clamped (upper)\n");
        //   printf("%f %f\n", S(0, 0), S(1, 1));
        // }
        S(0, 0) = min(S(0, 0), stretching.sigma_ub);
        S(1, 1) = min(S(1, 1), stretching.sigma_ub);
      }
      S(0, 0) = 1.0 / S(0, 0);
      S(1, 1) = 1.0 / S(1, 1);
      // printf("BP1");
      stretching.Dm = Vt.transpose() * S * U.transpose() * F;
      // printf("BP2");
      scene_ref.stretchings[sid] = stretching;
    }
  }
}

__global__ void UpdateBendingPlasticity(SceneRef scene_ref) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  if (bid < scene_ref.num_bending) {
    ElementBending bending = scene_ref.bendings[bid];
    constexpr float pi = 3.14159265358979323846;
    if (bending.elastic_limit < pi) {
      uint32_t a = scene_ref.bending_indices[bid * 4 + 0];
      uint32_t b = scene_ref.bending_indices[bid * 4 + 1];
      uint32_t c = scene_ref.bending_indices[bid * 4 + 2];
      uint32_t d = scene_ref.bending_indices[bid * 4 + 3];
      Matrix<float, 3, 4> X;
      X << scene_ref.x[a], scene_ref.x[b], scene_ref.x[c], scene_ref.x[d];
      DihedralAngle<float> dihedral_angle;
      float theta = dihedral_angle(X).value();
      float theta_rest = bending.theta_rest;

      // limit difference between theta and theta_rest to bending.bending_bound
      float diff = theta_rest - theta;
      if (diff > pi) {
        diff -= 2.0 * pi;
      }
      if (diff < -pi) {
        diff += 2.0 * pi;
      }
      if (diff < 0.0) {
        diff = max(diff, -bending.elastic_limit);
      }
      if (diff > 0.0) {
        diff = min(diff, bending.elastic_limit);
      }

      theta_rest = theta + diff;
      if (theta_rest < -pi) {
        theta_rest += 2.0 * pi;
      }
      if (theta_rest > pi) {
        theta_rest -= 2.0 * pi;
      }
      bending.theta_rest = theta_rest;
      scene_ref.bendings[bid] = bending;
    }
  }
}

__global__ void UpdateVelocity(SceneRef scene_ref, float dt) {
  int pid = threadIdx.x + blockIdx.x * blockDim.x;
  if (pid < scene_ref.num_particle) {
    scene_ref.v[pid] = (scene_ref.x[pid] - scene_ref.x_prev[pid]) / dt;
  }
}

void SceneDevice::Update(SceneDevice &scene, float dt) {
  DeviceClock clk;
  SceneRef scene_ref = scene;
  scene.x_prev_ = scene.x_;
  InitializeSolver<<<DEFAULT_DISPATCH_SIZE(scene_ref.num_particle), 0, scene.stream_>>>(
      scene_ref, Vector3<float>{0.0, -9.8, 0.0}, dt);
  clk.Record("Initialize Solver");

  const int num_vbd_iterations_ = 40;
  for (int iter = 0; iter < num_vbd_iterations_; iter++) {
    for (int c = 0; c < scene.particle_directory_host_.first.size(); c++) {
      SolveVBDParticlePosition<<<DEFAULT_DISPATCH_SIZE(scene.particle_directory_host_.count[c])>>>(
          scene_ref, scene.particle_directory_host_.positions.data() + scene.particle_directory_host_.first[c],
          scene.particle_directory_host_.count[c], dt);
    }
  }
  clk.Record("Solve VBD");

  UpdateVelocity<<<DEFAULT_DISPATCH_SIZE(scene_ref.num_particle), 0, scene.stream_>>>(scene_ref, dt);
  clk.Record("Update Velocity");

  UpdateStretchingPlasticity<<<DEFAULT_DISPATCH_SIZE(scene_ref.num_stretching)>>>(scene_ref);
  clk.Record("Update stretching plasticity");

  UpdateBendingPlasticity<<<DEFAULT_DISPATCH_SIZE(scene_ref.num_bending)>>>(scene_ref);
  clk.Record("Update bending plasticity");

  clk.Finish();
  printf("particles: %d, stretchings: %d, bendings: %d\n", scene_ref.num_particle, scene_ref.num_stretching,
         scene_ref.num_bending);
}

void SceneDevice::UpdateBatch(const std::vector<SceneDevice *> &scenes, float dt) {
  DeviceClock clk;
  std::vector<SceneRef> scene_refs(scenes.size());
  for (int i = 0; i < scenes.size(); i++) {
    scene_refs[i] = *scenes[i];
    scenes[i]->x_prev_ = scenes[i]->x_;
    InitializeSolver<<<DEFAULT_DISPATCH_SIZE(scene_refs[i].num_particle), 0, scenes[i]->stream_>>>(
        scene_refs[i], Vector3<float>{0.0, -9.8, 0.0}, dt);
  }
  clk.Record("Initialize Solver");

  const int num_vbd_iterations_ = 20;
  for (int i = 0; i < scenes.size(); i++) {
    for (int iter = 0; iter < num_vbd_iterations_; iter++) {
      for (int c = 0; c < scenes[i]->particle_directory_host_.first.size(); c++) {
        SolveVBDParticlePosition<<<DEFAULT_DISPATCH_SIZE(scenes[i]->particle_directory_host_.count[c]), 0,
                                   scenes[i]->stream_>>>(
            scene_refs[i],
            scenes[i]->particle_directory_.positions.data().get() + scenes[i]->particle_directory_host_.first[c],
            scenes[i]->particle_directory_host_.count[c], dt);
      }
    }
  }
  clk.Record("Solve VBD");

  for (int i = 0; i < scenes.size(); i++) {
    UpdateVelocity<<<DEFAULT_DISPATCH_SIZE(scene_refs[i].num_particle), 0, scenes[i]->stream_>>>(scene_refs[i], dt);
  }
  clk.Record("Update Velocity");

  for (int i = 0; i < scenes.size(); i++) {
    UpdateStretchingPlasticity<<<DEFAULT_DISPATCH_SIZE(scene_refs[i].num_stretching)>>>(scene_refs[i]);
  }
  clk.Record("Update stretching plasticity");

  for (int i = 0; i < scenes.size(); i++) {
    UpdateBendingPlasticity<<<DEFAULT_DISPATCH_SIZE(scene_refs[i].num_bending)>>>(scene_refs[i]);
  }
  clk.Record("Update bending plasticity");
  clk.Finish();
}

}  // namespace snow_mount::solver
#endif
