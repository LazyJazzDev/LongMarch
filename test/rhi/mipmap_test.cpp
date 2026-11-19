#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include "grassland/rhi/rhi.h"

using namespace grassland::rhi;

bool Check(bool value, const char *operation) {
  if (!value)
    std::cerr << operation << " failed\n";
  return value;
}

int main() {
  BackendAPI backend = BACKEND_API_DEFAULT;
  if (const char *name = std::getenv("RHI_TEST_BACKEND")) {
    if (std::string_view(name) == "vulkan")
      backend = BACKEND_API_VULKAN;
    else if (std::string_view(name) == "d3d12")
      backend = BACKEND_API_D3D12;
    else if (std::string_view(name) == "metal")
      backend = BACKEND_API_METAL;
    else
      return 1;
  }
  ObjPtr<ICore> core = CreateCore(backend, {});
  ObjPtr<IDevice> device = core ? core->CreateDeviceByAutoSelection() : nullptr;
  if (!Check(bool(device), "Create device"))
    return 1;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IImage> texture = device->CreateImage({4, 2, IMAGE_FORMAT_R8G8B8A8_UNORM, 3});
  ObjPtr<IImage> output = device->CreateImage({4, 1, IMAGE_FORMAT_R8G8B8A8_UNORM});
  if (!Check(bool(texture) && bool(output), "Create mip texture"))
    return 1;
  if (!Check(!ObjPtr<IImage>(device->CreateImage({4, 2, IMAGE_FORMAT_R8G8B8A8_UNORM, 4})), "Reject excessive levels") ||
      !Check(!ObjPtr<IImage>(device->CreateImage({4, 2, IMAGE_FORMAT_R8G8B8A8_UNORM, 0})), "Reject zero levels"))
    return 1;
  ObjPtr<ICommandContext> upload = queue->CreateCommandContext();
  for (uint32_t level = 0; level < 3; ++level) {
    const uint32_t width = 4 >> level, height = level == 0 ? 2 : 1, pitch = width * 4 + 8;
    std::vector<uint8_t> pixels(pitch * height, 37);
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = 0; x < width; ++x) {
        for (int channel = 0; channel < 3; ++channel)
          pixels[y * pitch + x * 4 + channel] = channel == int(level) ? 255 : 0;
        pixels[y * pitch + x * 4 + 3] = 255;
      }
    if (!Check(upload->CmdUploadImage(texture.Get(), pixels.data(), pitch, level), "Upload mip"))
      return 1;
  }
  ObjPtr<ISubmissionToken> uploaded = upload->Submit();
  if (!Check(uploaded && uploaded->WaitHost(), "Wait upload"))
    return 1;
  std::array<uint8_t, 16> dummy{};
  ObjPtr<ICommandContext> invalid = queue->CreateCommandContext();
  if (!Check(!invalid->CmdUploadImage(texture.Get(), dummy.data(), 16, 3), "Reject invalid mip index"))
    return 1;
  invalid = queue->CreateCommandContext();
  if (!Check(!invalid->CmdUploadImage(texture.Get(), dummy.data(), 3, 2), "Reject short mip row"))
    return 1;

  BindingLayoutDesc layout{
      {{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  PipelineLayoutDesc pipeline{{layout}};
  const std::string shader = R"(
RHI_TEXTURE2D(0,0,float4,image);
RHI_SAMPLER(0,1,filterSampler);
struct Pixel { float4 position:SV_Position; };
Pixel vs(uint id:SV_VertexID) {
  Pixel p; p.position=float4(id==2?3.:-1.,id==1?3.:-1.,0,1); return p;
}
float4 ps(Pixel p):SV_Target0 {
  float level=p.position.x<1.?0.:p.position.x<2.?1.:p.position.x<3.?2.:.5;
  return image.SampleLevel(filterSampler,float2(.5,.5),level);
}
)";
  ObjPtr<IShader> vs = device->CreateShaderFromSource(shader.c_str(), "vs", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> ps = device->CreateShaderFromSource(shader.c_str(), "ps", SHADER_STAGE_FRAGMENT);
  ProgramDesc programDesc;
  programDesc.pipeline_layout = pipeline;
  programDesc.vertex_shader = vs.Get();
  programDesc.fragment_shader = ps.Get();
  programDesc.color_formats = {IMAGE_FORMAT_R8G8B8A8_UNORM};
  programDesc.rasterizer.cull_mode = CULL_MODE_NONE;
  ObjPtr<IProgram> program = device->CreateProgram(programDesc);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  if (!Check(program && group && group->BindResource(0, texture.Get()), "Create sample program"))
    return 1;
  SamplerDesc samplerDesc;
  auto sample = [&](const SamplerDesc &desc, const std::array<std::array<int, 3>, 4> &expected) {
    ObjPtr<ISampler> sampler = device->CreateSampler(desc);
    if (!sampler || !group->BindResource(1, sampler.Get()))
      return false;
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    RenderingInfo rendering;
    rendering.color_attachments = {{output.Get(), ATTACHMENT_LOAD_OP_DISCARD, ATTACHMENT_STORE_OP_STORE, {}}};
    if (!commands->CmdBindProgram(program.Get()) || !commands->CmdBeginRendering(rendering) ||
        !commands->CmdSetViewport({0, 0, 4, 1, 0, 1}) || !commands->CmdSetScissor({0, 0, 4, 1}) ||
        !commands->CmdBindGroup(0, group.Get()) || !commands->CmdDraw(3) || !commands->CmdEndRendering())
      return false;
    ObjPtr<IDownloadToken> pixels = commands->CmdDownloadImage(output.Get());
    ObjPtr<ISubmissionToken> done = commands->Submit();
    std::array<uint8_t, 16> bytes{};
    if (!pixels || !done || !done->WaitHost() || !pixels->CopyTo(bytes.data(), bytes.size()))
      return false;
    for (size_t x = 0; x < 4; ++x)
      for (size_t channel = 0; channel < 3; ++channel)
        if (std::abs(int(bytes[x * 4 + channel]) - expected[x][channel]) > 2) {
          std::cerr << "pixel " << x << " channel " << channel << " = " << int(bytes[x * 4 + channel]) << '\n';
          return false;
        }
    return true;
  };
  if (!Check(sample(samplerDesc, {{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {128, 128, 0}}}), "Trilinear mip sample"))
    return 1;
  samplerDesc.min_lod = samplerDesc.max_lod = 1;
  if (!Check(sample(samplerDesc, {{{0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}}}), "LOD clamp"))
    return 1;
  samplerDesc = {};
  samplerDesc.max_anisotropy = 4;
  ObjPtr<ISampler> anisotropic = device->CreateSampler(samplerDesc);
  if (!Check(bool(anisotropic), "Create four-tap anisotropic sampler"))
    return 1;
  samplerDesc.max_anisotropy = 0;
  if (!Check(!ObjPtr<ISampler>(device->CreateSampler(samplerDesc)), "Reject zero anisotropy"))
    return 1;
  std::cout << "Mip upload, sampled chain, trilinear filtering, LOD clamp, anisotropy and invalid inputs verified\n";
  return 0;
}
