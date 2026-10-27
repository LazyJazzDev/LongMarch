#pragma once
#include "../../module.h"

class NBodyModule final : public ModuleBase {
 public:
  NBodyModule(grassland::rhix::Core core, grassland::rhix::Device device);
  ~NBodyModule() override;

 private:
  int OnInit(const ModuleOptions &options) override;
  int OnUpdate() override;
  int OnRender() override;
  int OnTerminate() override;

  bool ResetParticles(grassland::rhix::CommandContext &upload);
  bool UpdateSpeeds();

  struct State;
  std::unique_ptr<State> state_;
};
