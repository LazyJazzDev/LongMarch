#include "grassland/rhi/backend/backend_mip_generator.h"

#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi::backend {
ObjectPtr<IComputeProgram> CreateMipGenerationProgram(IDevice *device) {
  if (!device)
    return {};
  static const char source[] = R"(
      RHI_TEXTURE2D(0,0,float4,sourceImage);
      RHI_RW_TEXTURE2D(0,1,float4,targetImage);
      RHI_SAMPLER(0,2,linearSampler);
      [numthreads(8,8,1)] void Generate(uint3 id : SV_DispatchThreadID) {
        uint width,height;
        targetImage.GetDimensions(width,height);
        if (id.x < width && id.y < height)
          targetImage[id.xy] = sourceImage.SampleLevel(linearSampler,
                                  (float2(id.xy)+0.5)/float2(width,height),0.0);
      }
    )";
  auto shader = ObjectPtr<IShader>::Adopt(device->CreateShaderFromSource(source, "Generate", SHADER_STAGE_COMPUTE));
  if (!shader)
    return {};
  ComputeProgramDesc pipeline;
  pipeline.compute_shader = shader.Get();
  pipeline.pipeline_layout.groups = {{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                                       {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                                       {2, BINDING_SAMPLER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  return ObjectPtr<IComputeProgram>::Adopt(device->CreateComputeProgram(pipeline));
}

bool GenerateMipChain(ICommandContext *commands,
                      IDevice *device,
                      IImage *image,
                      uint32_t base_mip,
                      uint32_t mip_count,
                      ObjectPtr<IComputeProgram> &program,
                      ObjectPtr<ISampler> &sampler) {
  const auto &desc = image->GetDesc();
  if (base_mip >= desc.mip_levels)
    return false;
  if (!mip_count)
    mip_count = desc.mip_levels - base_mip;
  if (mip_count > desc.mip_levels - base_mip)
    return false;
  if (mip_count == 1)
    return true;
  if (!program)
    program = CreateMipGenerationProgram(device);
  if (!program)
    return false;
  if (!sampler) {
    SamplerDesc sampling{SAMPLER_FILTER_LINEAR, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_CLAMP_TO_EDGE};
    sampling.max_lod = 0;
    sampler = ObjectPtr<ISampler>::Adopt(device->CreateSampler(sampling));
    if (!sampler)
      return false;
  }
  auto group = BindingGroup(device->CreateBindingGroup(program->GetDesc().pipeline_layout.groups[0]));
  if (!group || !group.BindResource(2, sampler.Get()) || !commands->CmdBindComputeProgram(program.Get()))
    return false;
  for (uint32_t mip = base_mip + 1; mip < base_mip + mip_count; ++mip) {
    ImageDesc level;
    if (!ImageMipDesc(desc, mip, level) || !group.BindResource(0, ImageView{image, mip - 1, 1}) ||
        !group.BindResource(1, ImageView{image, mip, 1}) || !commands->CmdBindGroup(0, group.Get()) ||
        !commands->CmdDispatch((level.width + 7) / 8, (level.height + 7) / 8, 1))
      return false;
  }
  return true;
}
}  // namespace grassland::rhi::backend
