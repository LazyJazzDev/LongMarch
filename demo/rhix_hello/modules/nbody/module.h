#pragma once
#include <array>
#include <chrono>
#include <glm/glm.hpp>
#include <random>

#include "../../module.h"
#include "grassland/util/fps_counter.h"

class NBodyModule final : public ModuleBase {
 public:
  using ModuleBase::ModuleBase;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  bool ResetParticles(grassland::rhix::CommandContext &upload);
  bool UpdateSpeeds();

  bool initialized_{};
  bool headless_{};
  std::string capture_directory_{};
  grassland::rhi::BackendAPI api_{};
  grassland::rhix::Device device_{};
  grassland::rhix::DownloadToken last_frame_{};
  grassland::rhi::ImageDesc last_frame_desc_{};
  grassland::rhi::ColorSpace last_frame_space_{};
  float last_frame_white_scale_{};
  grassland::rhi::PresentMode requested_{};
  bool present_mode_changed_{};
  bool running_{};
  bool reset_requested_{};
  float delta_t_{};
  int galaxy_count_{};
  glm::mat4 rotation_{};
  glm::dvec2 last_cursor_{};
  glm::dvec2 rotation_delta_{};
  bool cursor_initialized_{};
  bool dragging_{};
  std::string status_{};
  bool capture_keyboard_{};
  bool output_change_requested_{};
  bool show_gui_{};
  grassland::rhi::ColorSpace requested_space_{};
  std::vector<grassland::rhi::ColorSpace> spaces_{};
  grassland::rhix::Window window_{};
  grassland::rhix::Queue queue_{};
  std::array<grassland::rhix::Image, 3> targets_{};
  std::array<grassland::rhix::SubmissionToken, 3> pending_{};
  grassland::rhix::Program program_{};
  grassland::rhix::Image radiance_{};
  grassland::rhix::Image frame_image_{};
  std::string output_source_{};
  grassland::rhi::BindingLayoutDesc output_layout_{};
  grassland::rhix::Buffer output_uniform_{};
  grassland::rhix::BindingGroup output_group_{};
  grassland::rhix::Program output_program_{};
  grassland::rhix::ComputeProgram compute_{};
  uint64_t bytes_{};
  grassland::rhix::Buffer positions_{};
  grassland::rhix::Buffer velocities_{};
  grassland::rhix::Buffer next_positions_{};
  grassland::rhix::Buffer simulation_uniform_{};
  grassland::rhix::Buffer camera_uniform_{};
  grassland::rhix::BindingGroup simulation_group_{};
  grassland::rhix::BindingGroup camera_group_{};
  std::mt19937 random_{};
  grassland::rhix::DownloadToken velocity_download_{};
  std::vector<glm::vec3> downloaded_velocities_{};
  std::array<uint32_t, 100> speed_counts_{};
  std::array<float, 100> normalized_speeds_{};
  float maximum_speed_{};
  bool speeds_ready_{};
  grassland::FPSCounter fps_{};
  std::chrono::steady_clock::time_point benchmark_start_{};
};
