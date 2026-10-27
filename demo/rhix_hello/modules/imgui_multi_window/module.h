#pragma once
#include "../../module.h"

class ImGuiMultiWindowModule final : public ModuleBase {
 public:
  ImGuiMultiWindowModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~ImGuiMultiWindowModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
