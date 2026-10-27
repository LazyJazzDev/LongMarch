#pragma once
#include "../../module.h"

class ImGuiModule final : public ModuleBase {
 public:
  ImGuiModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~ImGuiModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
