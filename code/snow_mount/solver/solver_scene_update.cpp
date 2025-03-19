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

  UpdateVelocity<<<DEFAULT_DISPATCH_SIZE(scene_ref.num_particle), 0, scene.stream_>>>(scene_ref, dt);
  clk.Record("Update Velocity");
}

void SceneDevice::UpdateBatch(const std::vector<SceneDevice *> &scenes, float dt) {
}

}  // namespace snow_mount::solver
#endif
