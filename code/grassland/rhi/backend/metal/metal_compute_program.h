#pragma once
#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalComputeProgram,
                        grassland::rhi::backend::metal::IMetalProgramBase,
                        "848b485d-2208-4d1d-86ae-cadee6182bfe")

namespace grassland::rhi::backend::metal {
class IMetalComputeProgram : public IMetalProgramBase {
 public:
  static rhi::IComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);
  virtual MTL::ComputePipelineState *NativePipeline() const = 0;
  virtual MTL::Size ThreadGroupSize() const = 0;
};

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalComputeProgramImpl final : public Object<rhi::IComputeProgram, IMetalComputeProgram> {
 public:
  static rhi::IComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);

  MTL::ComputePipelineState *NativePipeline() const;

  MTL::Size ThreadGroupSize() const;

  ~MetalComputeProgramImpl() override = default;

  const ComputeProgramDesc &GetDesc() const override;

  IMetalDevice *Owner() const override;

  const PipelineLayoutDesc &BindingLayout() const override;

 private:
  MetalComputeProgramImpl(IMetalDevice *device, const ComputeProgramDesc &desc);

  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;

  MTL::Size threads_{};

  void InitializeComputeProgram(IMetalDevice *device, const ComputeProgramDesc &desc);

  bool ValidateLayout() const;

  IMetalDevice *device_{};

  ComputeProgramDesc desc_{};

  ObjectPtr<IShader> shader_{};

  void InitializeMetalProgramBase(MTL::Device *device);

  NS::SharedPtr<MTL::Function> CompileFunction(const IShader *shader,
                                               const std::vector<InputAttributeDesc> &attributes = {},
                                               std::array<uint32_t, 3> *threads = nullptr) const;

  MTL::Device *native_device_{};
};
}  // namespace grassland::rhi::backend::metal
