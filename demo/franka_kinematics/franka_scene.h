#pragma once

#include <array>
#include <memory>

#include "sparkium/sparkium.h"

class FrankaScene {
 public:
  inline static constexpr std::array<float, 8> joint_lower{-2.7437f, -1.7837f, -2.9007f, -3.0421f,
                                                           -2.8065f, 0.5445f,  -3.0159f, 0.0f};
  inline static constexpr std::array<float, 8> joint_upper{2.7437f, 1.7837f, 2.9007f,  -0.1518f,
                                                           2.8065f, 4.5169f, 3.01599f, 0.04f};

  struct Settings {
    std::array<float, 8> joints{0.0f, 0.0f, 0.0f, -0.1518f, 0.0f, 0.5445f, 0.0f, 0.0f};
    bool ray_tracing{true};
  };

  static std::unique_ptr<FrankaScene> Create(grassland::rhi::IDevice *device, int width = 1024, int height = 512);
  ~FrankaScene();
  bool Configure(const Settings &settings);
  sparkium::Film *GetFilm();
  grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> Submit(int samples,
                                                                  grassland::rhi::IImage *output,
                                                                  bool hdr,
                                                                  const grassland::rhi::SubmitInfo &dependencies);

 private:
  struct State;
  explicit FrankaScene(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};
