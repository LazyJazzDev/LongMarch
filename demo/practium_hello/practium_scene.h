#pragma once
#include <cstdint>
#include <memory>

#include "sparkium/sparkium.h"

class PractiumDemoScene {
 public:
  struct Settings {
    bool ray_tracing{}, pause{true}, day_light{true};
    uint64_t reset_sequence{};
  };

  static std::unique_ptr<PractiumDemoScene> Create(grassland::rhi::IDevice *device, int width = 640, int height = 480);
  ~PractiumDemoScene();
  bool Configure(const Settings &settings);
  sparkium::Film *GetFilm();
  grassland::rhi::ObjPtr<grassland::rhi::ISubmissionToken> Submit(int samples,
                                                                  grassland::rhi::IImage *output,
                                                                  bool hdr,
                                                                  const grassland::rhi::SubmitInfo &dependencies);

 private:
  struct State;
  explicit PractiumDemoScene(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};
