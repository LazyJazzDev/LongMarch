#pragma once
#include "../../module.h"

class RayTracingModule final : public ModuleBase {
 public:
  RayTracingModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                   grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  ~RayTracingModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
