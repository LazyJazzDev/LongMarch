#pragma once
#include <filesystem>
#include <memory>
#include <string>

#include "grassland/graphics/graphics.h"

// Window-independent adapters for the desktop graphics demos. Rendering and NBody
// use the original HLSL through the same graphics API and Metal shader cache.
class DemoSession {
 public:
  DemoSession(const std::filesystem::path &resources, const std::string &demo, bool prepare = false);
  ~DemoSession();
  void Resize(int width, int height);
  void Configure(int particles, int galaxies, float delta_time, bool simulate, float yaw, float pitch, int reset);
  void Render();
  grassland::graphics::Core *Core() const {
    return core_.get();
  }
  grassland::graphics::Image *Image() const {
    return color_.get();
  }
  double GPUMilliseconds() const {
    return gpu_ms_;
  }
  int Particles() const {
    return particles_;
  }
  std::vector<glm::vec3> Positions() const;
  static const std::vector<std::string> &Names();

 private:
  void InitializeRaster();
  void InitializeNBody();
  void ResetParticles();
  std::unique_ptr<grassland::graphics::Buffer> Buffer(const void *data, size_t size);
  std::string demo_;
  std::unique_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Image> color_, depth_, texture_;
  std::unique_ptr<grassland::graphics::Sampler> sampler_;
  std::unique_ptr<grassland::graphics::Buffer> vertices_, indices_, uniform_, settings_, positions_, velocities_,
      next_positions_;
  std::unique_ptr<grassland::graphics::Shader> vertex_, fragment_, compute_;
  std::unique_ptr<grassland::graphics::Program> program_;
  std::unique_ptr<grassland::graphics::ComputeProgram> compute_program_;
  int width_ = 1280, height_ = 720, index_count_ = 3;
  int particles_ = 4096, galaxies_ = 10, reset_ = 0;
  float delta_time_ = 0.03f, theta_ = 0, yaw_ = 0, pitch_ = 0;
  bool simulate_ = true;
  double gpu_ms_ = 0;
};
