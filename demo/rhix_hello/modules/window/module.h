#pragma once
#include "../../module.h"

class WindowModule final : public ModuleBase {
 public:
  WindowModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~WindowModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
