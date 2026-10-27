#pragma once
#include "../../module.h"

class CubeModule final : public ModuleBase {
 public:
  CubeModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
             grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  ~CubeModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
