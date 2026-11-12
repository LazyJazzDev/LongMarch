#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalProgram,
                        grassland::rhi::backend::metal::IMetalProgramBase,
                        "26dfc710-560c-4dff-bfbd-2adc53cd3739")

namespace grassland::rhi::backend::metal {
class IMetalProgram : public IMetalProgramBase {
 public:
  static rhi::IProgram *Create(IMetalDevice *device, const ProgramDesc &desc);
  virtual MTL::RenderPipelineState *NativePipeline() const = 0;
  virtual MTL::DepthStencilState *NativeDepthStencil() const = 0;
};

}  // namespace grassland::rhi::backend::metal

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
