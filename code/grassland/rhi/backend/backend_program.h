#pragma once
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
class Device;

class Program : public Object<IProgram> {
 public:
  Program(Device *device, const ProgramDesc &desc);
  ~Program() override;
  const ProgramDesc &GetDesc() const override;
  bool IsFinalized() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  Device *device_;
  ProgramDesc desc_;
  bool finalized_{};

 private:
  ObjectPtr<IShader> vertex_shader_;
  ObjectPtr<IShader> fragment_shader_;
};
}  // namespace grassland::rhi::backend
