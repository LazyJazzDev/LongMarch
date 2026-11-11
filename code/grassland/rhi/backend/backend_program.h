#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Program;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Program)
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
class Device;

class Program : public Object<IProgram> {
 public:
  LONGMARCH_RHI_QUERY(Program, Object<IProgram>)

  Program(Device *device, const ProgramDesc &desc);
  ~Program() override;
  const ProgramDesc &GetDesc() const override;
  Device *Owner() const;

 protected:
  bool ValidateLayout() const;
  Device *device_;
  ProgramDesc desc_;

 private:
  ObjectPtr<IShader> vertex_shader_;
  ObjectPtr<IShader> fragment_shader_;
};
}  // namespace grassland::rhi::backend
