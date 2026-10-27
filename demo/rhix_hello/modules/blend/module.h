#pragma once
#include "../../module.h"

class BlendModule final : public ModuleBase {
 public:
  BlendModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~BlendModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
