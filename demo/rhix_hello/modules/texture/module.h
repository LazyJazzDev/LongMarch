#pragma once
#include "../../module.h"

class TextureModule final : public ModuleBase {
 public:
  TextureModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~TextureModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
