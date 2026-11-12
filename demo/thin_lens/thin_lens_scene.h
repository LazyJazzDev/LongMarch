#pragma once

#include "sparkium/sparkium.h"

// Owned exclusively by the rendering thread; controls cross threads as Settings.
class ThinLensScene {
 public:
  struct Settings {
    bool pinhole_mode{};
    float focus_distance{5.4f}, aperture_radius{0.22f};
    int aperture_blades{6};
    float aperture_rotation{}, aperture_ratio{1.0f}, geometry_offset{0.1f};
  };

  static std::unique_ptr<ThinLensScene> Create(grassland::rhi::IDevice *device, int width = 1100, int height = 700);
  bool Configure(const Settings &settings);
  sparkium::Film *GetFilm();
  grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> Submit(int samples,
                                                                  grassland::rhi::IImage *output,
                                                                  bool hdr,
                                                                  const grassland::rhi::SubmitInfo &dependencies);

 private:
  ThinLensScene(std::unique_ptr<sparkium::Core> core, int width, int height);
  std::unique_ptr<sparkium::Core> core_;
  sparkium::Scene scene_;
  sparkium::GeometryMesh sphere_, tile_;
  std::vector<std::unique_ptr<sparkium::Material>> materials_;
  std::vector<std::unique_ptr<sparkium::EntityGeometryMaterial>> entities_;
  sparkium::Film film_;
  sparkium::CameraThinLens lens_;
  sparkium::CameraPinhole pinhole_;
  sparkium::Renderer renderer_;
  sparkium::RenderPipeline pipeline_;
  bool pinhole_mode_{};
};
