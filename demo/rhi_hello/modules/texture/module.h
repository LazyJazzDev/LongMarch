#pragma once
#include "../../module.h"

class TextureModule final : public ModuleBase {
 public:
  TextureModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  ~TextureModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  struct State;
  std::unique_ptr<State> state_;
};
