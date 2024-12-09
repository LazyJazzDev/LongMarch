#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Core {
 public:
  struct Settings {
    int frames_in_flight{2};
    bool enable_debug{false};
    bool enable_ray_tracing{false};
  };
  Core(const Settings &settings);

  virtual ~Core() = default;

  virtual int CreateBuffer(size_t size,
                           BufferType type,
                           double_ptr<Buffer> pp_buffer) = 0;

  virtual int CreateImage(int width,
                          int height,
                          ImageFormat format,
                          double_ptr<Image> pp_image) = 0;

  int FramesInFlight() const;

  bool DebugEnabled() const;

  bool RayTracingEnabled() const;

 private:
  Settings settings_;
};

int CreateCore(BackendAPI api,
               const Core::Settings &settings,
               double_ptr<Core> pp_core);

}  // namespace grassland::graphics
