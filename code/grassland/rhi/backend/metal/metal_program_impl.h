#pragma once
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalProgramImpl final : public Object<rhi::IProgram, IMetalProgram> {
 public:
  static rhi::IProgram *Create(IMetalDevice *device, const ProgramDesc &desc);

  MTL::RenderPipelineState *NativePipeline() const;

  MTL::DepthStencilState *NativeDepthStencil() const;

  static constexpr uint32_t VertexBufferBase = 8;

  ~MetalProgramImpl() override = default;

  const ProgramDesc &GetDesc() const override;

  IMetalDevice *Owner() const override;

  const PipelineLayoutDesc &BindingLayout() const override;

 private:
  MetalProgramImpl(IMetalDevice *device, const ProgramDesc &desc);

  NS::SharedPtr<MTL::RenderPipelineState> pipeline_;

  NS::SharedPtr<MTL::DepthStencilState> depth_stencil_;

  void InitializeProgram(IMetalDevice *device, const ProgramDesc &desc);

  bool ValidateLayout() const;

  IMetalDevice *device_{};

  ProgramDesc desc_{};

  ObjectPtr<IShader> vertex_shader_{};

  ObjectPtr<IShader> fragment_shader_{};

  void InitializeMetalProgramBase(MTL::Device *device);

  NS::SharedPtr<MTL::Function> CompileFunction(const IShader *shader,
                                               const std::vector<InputAttributeDesc> &attributes = {},
                                               std::array<uint32_t, 3> *threads = nullptr) const;

  MTL::Device *native_device_{};
};
}  // namespace grassland::rhi::backend::metal
