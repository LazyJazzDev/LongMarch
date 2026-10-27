#pragma once
#include "../../module.h"

class TriangleModule final : public ModuleBase {
 public:
  TriangleModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                 grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  ~TriangleModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
