#pragma once
#include "glm/gtc/matrix_transform.hpp"
#include "long_march.h"
#include "params.h"
#include "random"

void UpdateStep(glm::vec3 *positions, glm::vec3 *velocities, glm::vec3 *positions_new, int n_particles, float delta_t);

using namespace long_march;

struct GlobalUniformObject {
  glm::mat4 world_to_screen;
  glm::mat4 camera_to_world;
  float particle_size;
  int hdr;
};

class NBodyCUDA {
 public:
  struct Settings {
    int n_particles{NUM_PARTICLE};
    bool headless{false};
    int num_step{200};
    int device_id{0};
  };
  explicit NBodyCUDA(Settings settings);
  ~NBodyCUDA();
  void Run();

 private:
  void OnInit();
  void OnUpdate();
  void OnRender();
  void OnClose();

  void BuildRenderNode();

  void UpdateParticles();
  void UpdateRenderAssets();
  void UpdateImGui();

  float RandomFloat();
  glm::vec3 RandomOnSphere();
  glm::vec3 RandomInSphere();

  void ResetParticles();

  std::unique_ptr<graphics::Core> core_;
  std::unique_ptr<graphics::Window> window_;
  std::unique_ptr<graphics::Buffer> global_uniform_buffer_;
  std::unique_ptr<graphics::CUDABuffer> particles_pos_;
  std::unique_ptr<graphics::Image> frame_image_;
  std::unique_ptr<graphics::Shader> vertex_shader_;
  std::unique_ptr<graphics::Shader> fragment_shader_;
  std::unique_ptr<graphics::Program> program_;

  std::unique_ptr<graphics::Shader> hdr_vertex_shader_;
  std::unique_ptr<graphics::Shader> hdr_fragment_shader_;
  std::unique_ptr<graphics::Program> hdr_program_;

  glm::vec3 *positions_;
  glm::vec3 *velocities_;
  glm::vec3 *positions_new_;

  int n_particles_;
  std::mt19937 random_device_{uint32_t(std::time(nullptr))};
  glm::mat4 rotation{1.0f};
  bool hdr_{false};
  bool step_{true};
  float delta_t_{DELTA_T};
  int galaxy_number_{10};

  bool headless_{false};
  int num_step_{1024};
};
