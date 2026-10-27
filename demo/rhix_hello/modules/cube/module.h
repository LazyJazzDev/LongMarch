#pragma once
#include "../../module.h"

class CubeModule final : public ModuleBase {
 public:
  CubeModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~CubeModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
