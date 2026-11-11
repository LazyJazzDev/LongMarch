#pragma once
#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalComputeProgramImpl final : public Object<IMetalComputeProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IComputeProgram,
                            IMetalComputeProgram,
                            IBackendComputeProgram,
                            IComputeProgram,
                            IMetalProgramBase,
                            IObject)

  static IMetalComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);

  MTL::ComputePipelineState *NativePipeline() const;

  MTL::Size ThreadGroupSize() const;

  ~MetalComputeProgramImpl() override = default;

  const ComputeProgramDesc &GetDesc() const override;

  IBackendDevice *Owner() const override;

  const PipelineLayoutDesc &BindingLayout() const override;

 private:
  MetalComputeProgramImpl(IMetalDevice *device, const ComputeProgramDesc &desc);

  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;

  MTL::Size threads_{};

  void InitializeComputeProgram(IBackendDevice *device, const ComputeProgramDesc &desc);

  bool ValidateLayout() const;

  IBackendDevice *device_{};

  ComputeProgramDesc desc_{};

  ObjectPtr<IShader> shader_{};

  void InitializeMetalProgramBase(MTL::Device *device);

  NS::SharedPtr<MTL::Function> CompileFunction(const IShader *shader,
                                               const std::vector<InputAttributeDesc> &attributes = {},
                                               std::array<uint32_t, 3> *threads = nullptr) const;

  MTL::Device *native_device_{};
};
}  // namespace grassland::rhi::backend::metal
