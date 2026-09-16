#pragma once
#include "grassland/graphics/shader_cache.h"
#include "sparkium/core/core.h"
#include "sparkium/scene_io/json_scene.h"

// All methods, construction and destruction run on one rendering queue.
class RenderSession {
 public:
  RenderSession(const std::filesystem::path &resources,
                const std::string &scene,
                int max_dimension,
                bool prepare = false);
  ~RenderSession();
  std::vector<uint8_t> Step();
  int Width() const;
  int Height() const;
  int Samples() const;
  std::string Device() const;

 private:
  std::unique_ptr<grassland::graphics::Core> graphics_;
  std::unique_ptr<sparkium::Core> core_;
  std::unique_ptr<sparkium::JsonScene> scene_;
  std::unique_ptr<grassland::graphics::Image> image_;
};
