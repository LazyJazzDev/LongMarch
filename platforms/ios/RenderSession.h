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
                bool prepare = false,
                double aspect_ratio = 0.0,
                grassland::graphics::BackendAPI backend = grassland::graphics::BACKEND_API_DEFAULT,
                bool allow_compute_fallback = false,
                bool force_compute_fallback = false);
  ~RenderSession();
  std::vector<uint8_t> Step();
  void Render();
  grassland::graphics::Image *Develop(bool hdr, float exposure = 0);

  grassland::graphics::Core *Graphics() const {
    return graphics_.get();
  }

  bool ComputeFallback() const {
    return compute_fallback_;
  }

  std::vector<uint8_t> Display(bool hdr, float exposure = 0);
  int Width() const;
  int Height() const;
  int Samples() const;
  int MaxBounces() const;
  // HDR development is linear only without Filmic or artistic grading.
  bool LinearHDRLook() const;
  void ResetFilm();
  std::string Device() const;

 private:
  std::unique_ptr<grassland::graphics::Core> graphics_;
  std::unique_ptr<sparkium::Core> core_;
  std::unique_ptr<sparkium::JsonScene> scene_;
  std::unique_ptr<grassland::graphics::Image> image_, hdr_image_;
  float scene_exposure_ = 0;
  bool compute_fallback_ = false;
};
