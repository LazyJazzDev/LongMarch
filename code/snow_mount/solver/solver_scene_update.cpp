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
      scene_ref.rigid_objects[i].mesh_sdf.SDF(x, &sdf, &jacobian, &hessian);
      Vector3<float> contact_point = x - sdf * jacobian;
      sdf -= 0.018;
      if (sdf < 0.0) {
        // printf("pid: %d\nx: %f %f %f\nsdf: %f\njacobian: %f %f %f\nmesh_t: %f %f %f\n", pid, x[0], x[1], x[2], sdf,
        //        jacobian[0], jacobian[1], jacobian[2], scene_ref.rigid_objects[i].mesh_sdf.translation[0],
        //        scene_ref.rigid_objects[i].mesh_sdf.translation[1],
        //        scene_ref.rigid_objects[i].mesh_sdf.translation[2]);
        float k_stiffness = scene_ref.rigid_objects[i].stiffness * m;
        float force_mag = -2.0 * k_stiffness * sdf;
        Matrix3<float> partial_H =
            2.0 * k_stiffness * jacobian * jacobian.transpose() + 2.0 * k_stiffness * sdf * hessian;
        f -= 2.0 * k_stiffness * sdf * jacobian + partial_H * (x - x_prev) * K_DAMPING;
        H += partial_H + partial_H * K_DAMPING;

        Vector3<float> velocity_component = rel_vel - scene_ref.rigid_objects[i].v;
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
    // if (tid == 0) {
    //   printf("f: %f %f %f\n", f[0], f[1], f[2]);
    //   printf("H=\n%f %f %f\n", H(0, 0), H(0, 1), H(0, 2));
    //   printf("%f %f %f\n", H(1, 0), H(1, 1), H(1, 2));
    //   printf("%f %f %f\n", H(2, 0), H(2, 1), H(2, 2));
    // }
    Vector3<float> delta_x = H.inverse() * f;
    scene_ref.x[pidx] += delta_x;
  }
}

__global__ void SolveVBDParticlePositionLaunch(DirectoryRef directory_ref,
                                               int num_color,
                                               SceneRef scene_ref,
                                               float dt) {
  for (int c = 0; c < num_color; c++) {
    SolveVBDParticlePosition<<<DEFAULT_DISPATCH_SIZE(directory_ref.count[c])>>>(
        scene_ref, directory_ref.positions + directory_ref.first[c], directory_ref.count[c], dt);
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
    // ApplyFrictionForcesInLoop<<<DEFAULT_DISPATCH_SIZE(num_particle)>>>(scene_device_ref, dt_);
    SolveVBDParticlePositionLaunch<<<1, 1, 0, scene.stream_>>>(scene.particle_directory_,
                                                               scene.particle_directory_.first.size(), scene_ref, dt);
  }
  clk.Record("Solve VBD");

  UpdateVelocity<<<DEFAULT_DISPATCH_SIZE(scene_ref.num_particle), 0, scene.stream_>>>(scene_ref, dt);
  clk.Record("Update Velocity");
  clk.Finish();
  printf("particles: %d, stretchings: %d, bendings: %d\n", scene_ref.num_particle, scene_ref.num_stretching,
         scene_ref.num_bending);
}

void SceneDevice::UpdateBatch(const std::vector<SceneDevice *> &scenes, float dt) {
}

}  // namespace snow_mount::solver
#endif
