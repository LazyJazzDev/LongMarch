#pragma once
#include "../../module.h"

class NBodyModule final : public ModuleBase {
 public:
  NBodyModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
              grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  ~NBodyModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  bool ResetParticles(grassland::rhi::ICommandContext *upload);
  bool UpdateSpeeds();

  struct State;
  std::unique_ptr<State> state_;
};
