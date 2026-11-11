#pragma once
#include "grassland/rhi/backend/backend_program_state.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_program_base_state.h"

namespace grassland::rhi::backend::metal {
class MetalProgramImpl final : public Object<IMetalProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IProgram, IMetalProgram, Program, IProgram, MetalProgramBase, IObject)
  static IMetalProgram *Create(IMetalDevice *device, const ProgramDesc &desc);
  MTL::RenderPipelineState *NativePipeline() const;
  MTL::DepthStencilState *NativeDepthStencil() const;
  static constexpr uint32_t VertexBufferBase = 8;

 private:
  MetalProgramImpl(IMetalDevice *device, const ProgramDesc &desc);
  NS::SharedPtr<MTL::RenderPipelineState> pipeline_;
  NS::SharedPtr<MTL::DepthStencilState> depth_stencil_;

 public:
  ~MetalProgramImpl() override {
  }

 private:
  ProgramState state_Program_;

  ProgramState &StateProgram() override {
    return state_Program_;
  }

  const ProgramState &StateProgram() const override {
    return state_Program_;
  }

 private:
  MetalProgramBaseState state_MetalProgramBase_;

  MetalProgramBaseState &StateMetalProgramBase() override {
    return state_MetalProgramBase_;
  }

  const MetalProgramBaseState &StateMetalProgramBase() const override {
    return state_MetalProgramBase_;
  }
};
}  // namespace grassland::rhi::backend::metal
