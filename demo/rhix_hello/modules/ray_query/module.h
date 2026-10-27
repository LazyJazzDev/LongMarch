#pragma once
#include "../../module.h"

class RayQueryModule final : public ModuleBase {
 public:
  RayQueryModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~RayQueryModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
