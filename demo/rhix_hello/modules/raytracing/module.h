#pragma once
#include "../../module.h"

class RayTracingModule final : public ModuleBase {
 public:
  RayTracingModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~RayTracingModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
