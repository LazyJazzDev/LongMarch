#pragma once
#include "../../module.h"

class ImGuiModule final : public ModuleBase {
 public:
  ImGuiModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
              grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  ~ImGuiModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
