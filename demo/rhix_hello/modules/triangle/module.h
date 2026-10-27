#pragma once
#include "../../module.h"

class TriangleModule final : public ModuleBase {
 public:
  TriangleModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~TriangleModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
