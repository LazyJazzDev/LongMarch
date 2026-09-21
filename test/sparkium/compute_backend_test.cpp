#include <gtest/gtest.h>
#include <long_march.h>

#include <array>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <random>

#include "sparkium/backend/cpu/cpu_bvh.h"

using namespace grassland;

namespace {
class ComputeBackendTest : public testing::TestWithParam<sparkium::RenderBackend> {
 protected:
  void SetUp() override {
    if (!sparkium::SupportBackend(GetParam()))
      GTEST_SKIP() << "backend not compiled";
    ASSERT_EQ(sparkium::CreateDevice(GetParam(), {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
    if (GetParam() == sparkium::RenderBackend::CPU)
      EXPECT_FALSE(core->DeviceRayTracingSupport());
    EXPECT_FALSE(core->DeviceRayQuerySupport());
    EXPECT_EQ(core->API(), GetParam());
  }

  std::unique_ptr<sparkium::backend::Device> core;
};

TEST_P(ComputeBackendTest, DescriptorArraysRangesAndIndependentEntryPoints) {
  // Both modules deliberately use Main. Their generated host entry helpers
  // must not interpose on one another. Array and cbuffer offsets differ.
  const std::string prefix = R"(
ByteAddressBuffer inputs[] : register(t0, space0);
struct Settings { uint count; uint value; };
ConstantBuffer<Settings> settings : register(b0, space1);
RWByteAddressBuffer output : register(u0, space2);
[numthreads(8,1,1)] void Main(uint3 id : SV_DispatchThreadID) {
  if (id.x < settings.count) output.Store(id.x*4,
)";
  std::unique_ptr<graphics::Shader> a, b;
  core->CreateShader(prefix + "inputs[0].Load(id.x*4)+settings.value); }", "Main", "cs_6_0", &a);
  core->CreateShader(prefix + "inputs[1].Load(id.x*4)*settings.value); }", "Main", "cs_6_0", &b);
  std::unique_ptr<graphics::Buffer> input, output, settings;
  core->CreateBuffer(76, graphics::BUFFER_TYPE_STATIC, &input);
  core->CreateBuffer(76, graphics::BUFFER_TYPE_STATIC, &output);
  core->CreateBuffer(264, graphics::BUFFER_TYPE_STATIC, &settings);
  uint32_t values[19];
  for (uint32_t i = 0; i < 19; ++i)
    values[i] = i + 3;
  input->UploadData(values, sizeof(values));
  uint32_t params[2] = {19, 7};
  settings->UploadData(params, sizeof(params), 256);
  for (auto *shader : {a.get(), b.get(), a.get()}) {
    std::unique_ptr<graphics::ComputeProgram> program;
    core->CreateComputeProgram(shader, &program);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 2);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    program->Finalize();
    std::unique_ptr<graphics::CommandContext> cmd;
    core->CreateCommandContext(&cmd);
    cmd->CmdBindComputeProgram(program.get());
    cmd->CmdBindResources(0, std::vector<graphics::Buffer *>{input.get(), input.get()}, graphics::BIND_POINT_COMPUTE);
    cmd->CmdBindResources(1, std::vector{settings->Range(256, 8)}, graphics::BIND_POINT_COMPUTE);
    cmd->CmdBindResources(2, std::vector<graphics::Buffer *>{output.get()}, graphics::BIND_POINT_COMPUTE);
    cmd->CmdDispatch(3, 1, 1);
    // Rebinding a compatible program must not discard descriptor state (the
    // shared light CDF's up/down-sweep changes programs without rebinding slot 0).
    std::unique_ptr<graphics::ComputeProgram> compatible;
    core->CreateComputeProgram(shader, &compatible);
    compatible->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 2);
    compatible->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    compatible->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    compatible->Finalize();
    cmd->CmdBindComputeProgram(compatible.get());
    cmd->CmdDispatch(3, 1, 1);
    core->SubmitCommandContext(cmd.get());
    uint32_t actual[19];
    output->DownloadData(actual, sizeof(actual));
    for (int i = 0; i < 19; ++i)
      EXPECT_EQ(actual[i], shader == b.get() ? values[i] * 7 : values[i] + 7);
  }
}

TEST_P(ComputeBackendTest, ThreeDimensionalInvocationIdsAndConstantBlock) {
  std::unique_ptr<graphics::Shader> shader;
  core->CreateShader(R"(
cbuffer Parameters : register(b0, space0) { uint scale; uint bias; };
RWByteAddressBuffer output : register(u0, space1);
[numthreads(2,3,2)] void Main(uint3 id : SV_DispatchThreadID,
                            uint3 local : SV_GroupThreadID,
                            uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
  uint index=id.x+10*(id.y+12*id.z);
  output.Store3(index*12,uint3(index+bias,(group.x+5*(group.y+4*group.z))*scale,
                             lane+100*(local.x+2*(local.y+3*local.z))));
})",
                     "Main", "cs_6_0", &shader);
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<graphics::Buffer> params, output;
  core->CreateBuffer(8, graphics::BUFFER_TYPE_STATIC, &params);
  core->CreateBuffer(720 * 12, graphics::BUFFER_TYPE_STATIC, &output);
  for (unsigned bias : {7u, 19u}) {
    uint32_t settings[]{3, bias};
    params->UploadData(settings, sizeof(settings));
    std::unique_ptr<graphics::CommandContext> commands;
    core->CreateCommandContext(&commands);
    commands->CmdBindComputeProgram(program.get());
    commands->CmdBindResources(0, std::vector<graphics::Buffer *>{params.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, std::vector<graphics::Buffer *>{output.get()}, graphics::BIND_POINT_COMPUTE);
    commands->CmdDispatch(5, 4, 3);
    core->SubmitCommandContext(commands.get());
    std::array<uint32_t, 720 * 3> values;
    output->DownloadData(values.data(), sizeof(values));
    for (unsigned z = 0; z < 6; ++z)
      for (unsigned y = 0; y < 12; ++y)
        for (unsigned x = 0; x < 10; ++x) {
          unsigned index = x + 10 * (y + 12 * z);
          EXPECT_EQ(values[index * 3], index + bias);
          EXPECT_EQ(values[index * 3 + 1], (x / 2 + 5 * (y / 3 + 4 * (z / 2))) * 3);
          EXPECT_EQ(values[index * 3 + 2], (x % 2 + 2 * (y % 3 + 3 * (z % 2))) * 101);
        }
  }
}

TEST_P(ComputeBackendTest, TextureArraysFilteringAndPartialImageTransfers) {
  std::unique_ptr<graphics::Image> input, output;
  core->CreateImage(2, 2, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &input);
  core->CreateImage(7, 5, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output);
  std::array<uint8_t, 16> texels{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
  input->UploadData(texels.data());
  std::array<uint8_t, 4> partial{}, replacement{128, 64, 32, 255};
  input->DownloadData(partial.data(), {1, 1}, {1, 1});
  EXPECT_EQ(partial[0], 255);
  input->UploadData(replacement.data(), {1, 1}, {1, 1});
  input->DownloadData(partial.data(), {1, 1}, {1, 1});
  EXPECT_EQ(partial, replacement);
  input->UploadData(texels.data());
  std::unique_ptr<graphics::Shader> shader;
  core->CreateShader(R"(
Texture2D<float4> images[] : register(t0,space0);
SamplerState samplers[] : register(s0,space1);
RWTexture2D<float4> output : register(u0,space2);
[numthreads(8,8,1)] void Main(uint3 id : SV_DispatchThreadID) {
 uint w,h; output.GetDimensions(w,h);
 if(id.x<w && id.y<h) output[id.xy]=images[1].SampleLevel(samplers[0],float2(0.5,0.5),0);
})",
                     "Main", "cs_6_0", &shader);
  std::unique_ptr<graphics::Sampler> sampler;
  core->CreateSampler({graphics::FILTER_MODE_LINEAR}, &sampler);
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 2);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_SAMPLER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program->Finalize();
  std::unique_ptr<graphics::CommandContext> cmd;
  core->CreateCommandContext(&cmd);
  cmd->CmdBindComputeProgram(program.get());
  cmd->CmdBindResources(0, std::vector<graphics::Image *>{input.get(), input.get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdBindResources(1, std::vector<graphics::Sampler *>{sampler.get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdBindResources(2, std::vector<graphics::Image *>{output.get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdDispatch(1, 1, 1);
  core->SubmitCommandContext(cmd.get());
  std::array<float, 140> actual;
  output->DownloadData(actual.data());
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], i % 4 == 3 ? 1.0f : 0.5f, 1e-6f);
}

TEST_P(ComputeBackendTest, HDRFloat3AndNearestAddressModes) {
  std::unique_ptr<graphics::Image> input, output;
  core->CreateImage(2, 1, graphics::IMAGE_FORMAT_R32G32B32_SFLOAT, &input);
  core->CreateImage(4, 2, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output);
  const float texels[]{100, 101, 102, 200, 201, 202};
  input->UploadData(texels);
  std::unique_ptr<graphics::Shader> shader;
  core->CreateShader(R"(
Texture2D<float3> image : register(t0,space0);
SamplerState samplers[] : register(s0,space1);
RWTexture2D<float4> output : register(u0,space2);
[numthreads(4,2,1)] void Main(uint3 id : SV_DispatchThreadID) {
  float u=id.y==0 ? -0.25f : 1.25f;
  output[id.xy]=float4(image.SampleLevel(samplers[id.x],float2(u,0.5f),0),1);
})",
                     "Main", "cs_6_0", &shader);
  std::array<std::unique_ptr<graphics::Sampler>, 4> samplers;
  std::vector<graphics::Sampler *> bindings;
  for (int mode = 0; mode < 4; ++mode) {
    core->CreateSampler({graphics::FILTER_MODE_NEAREST, static_cast<graphics::AddressMode>(mode)}, &samplers[mode]);
    bindings.push_back(samplers[mode].get());
  }

  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_SAMPLER, 4);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program->Finalize();
  std::unique_ptr<graphics::CommandContext> cmd;
  core->CreateCommandContext(&cmd);
  cmd->CmdBindComputeProgram(program.get());
  cmd->CmdBindResources(0, std::vector<graphics::Image *>{input.get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdBindResources(1, bindings, graphics::BIND_POINT_COMPUTE);
  cmd->CmdBindResources(2, std::vector<graphics::Image *>{output.get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdDispatch(1, 1, 1);
  core->SubmitCommandContext(cmd.get());
  std::array<float, 32> actual;
  output->DownloadData(actual.data());
  const float expected[]{200, 100, 100, 0, 100, 200, 200, 0};
  for (int pixel = 0; pixel < 8; ++pixel) {
    for (int c = 0; c < 3; ++c)
      EXPECT_FLOAT_EQ(actual[pixel * 4 + c], expected[pixel] ? expected[pixel] + c : 0);
    EXPECT_FLOAT_EQ(actual[pixel * 4 + 3], 1);
  }

  EXPECT_THROW(input->DownloadData(actual.data(), {-1, 0}, {1, 1}), std::out_of_range);
  EXPECT_THROW(input->UploadData(texels, {2, 0}, {1, 1}), std::out_of_range);
}

TEST_P(ComputeBackendTest, FiniteLightClassification) {
  sparkium::Core renderer(core.get());
  auto vfs = renderer.GetShadersVFS();
  vfs.WriteFile("finite_test.hlsl", R"(
#include "common.hlsli"
#include "compute_contract.hlsli"
SP_RESOURCE(RWByteAddressBuffer, values, u0, 0);
#define VALUES SP_RESOURCE_ACCESS(RWByteAddressBuffer, values, 0)
SP_NUMTHREADS(1,1,1) void Main(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  float x = asfloat(VALUES.Load(id.x * 4));
  VALUES.Store(32 + id.x * 4, uint(isfinite(x)) |
      (uint(AllFinite(float3(x,x,x))) << 1) |
      (uint(AllFinite(float3(x,1,2))) << 2) |
      (uint(AllFinite(float3(1,x,2))) << 3) |
      (uint(AllFinite(float3(1,2,x))) << 4));
}
)");
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(core->CreateShader(vfs, "finite_test.hlsl", "Main", "cs_6_0", &shader), 0);
  std::unique_ptr<graphics::Buffer> values;
  core->CreateBuffer(64, graphics::BUFFER_TYPE_STATIC, &values);
  const uint32_t bits[]{0, 0x3f800000, 0xbf800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x00000001};
  values->UploadData(bits, sizeof(bits));
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<graphics::CommandContext> cmd;
  core->CreateCommandContext(&cmd);
  cmd->CmdBindComputeProgram(program.get());
  cmd->CmdBindResources(0, std::vector<graphics::Buffer *>{values.get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdDispatch(8, 1, 1);
  core->SubmitCommandContext(cmd.get());
  uint32_t actual[16];
  values->DownloadData(actual, sizeof(actual));
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(actual[8 + i], (i >= 4 && i <= 6) ? 0u : 31u) << "input " << i;
}

TEST_P(ComputeBackendTest, FrameAccumulationResetAndBackground) {
  sparkium::Core renderer(core.get());
  sparkium::Scene scene(&renderer);
  scene.settings.samples_per_dispatch = 3;
  scene.settings.max_bounces = 4;
  scene.settings.background_color = {0.1f, 0.2f, 0.3f};
  sparkium::Camera camera(&renderer, glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(45.0f), 17.0f / 13.0f);
  sparkium::Film film(&renderer, 17, 13);
  film.info.persistence = 0.9f;
  std::vector<glm::vec4> first(17 * 13), second(17 * 13), reset(17 * 13);
  renderer.Render(&scene, &camera, &film);
  film.GetRawImage()->DownloadData(first.data());
  renderer.Render(&scene, &camera, &film);
  film.GetRawImage()->DownloadData(second.data());
  EXPECT_EQ(film.info.accumulated_samples, 6);
  film.Reset();
  renderer.Render(&scene, &camera, &film);
  film.GetRawImage()->DownloadData(reset.data());
  EXPECT_EQ(film.info.accumulated_samples, 3);
  EXPECT_EQ(std::memcmp(first.data(), reset.data(), first.size() * sizeof(glm::vec4)), 0);
  // Film resolve intentionally truncates its effective sample count to int,
  // including with persistence. Check that the existing semantics are retained.
  float effective = 0;
  for (int i = 0; i < 3; ++i)
    effective = effective * 0.9f + 1;
  for (const auto &p : first)
    for (int c = 0; c < 3; ++c) {
      EXPECT_TRUE(std::isfinite(p[c]));
      EXPECT_NEAR(p[c], scene.settings.background_color[c] * effective / int(effective), 1e-5f);
    }
  EXPECT_THROW(renderer.Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RASTERIZATION), std::runtime_error);
  EXPECT_THROW(renderer.Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RAY_QUERY), std::runtime_error);
  if (core->DeviceRayTracingSupport()) {
    EXPECT_EQ(renderer.ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO), sparkium::RENDER_PIPELINE_RAY_TRACING);
    EXPECT_EQ(renderer.ResolveRenderPipeline(sparkium::RENDER_PIPELINE_RAY_TRACING),
              sparkium::RENDER_PIPELINE_RAY_TRACING);
    // Switching traversal must reset accumulation; Auto must keep the OptiX path.
    renderer.Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RT_FALLBACK);
    EXPECT_EQ(film.info.accumulated_samples, 3);
    renderer.Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RAY_TRACING);
    EXPECT_EQ(film.info.accumulated_samples, 3);
    renderer.Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_AUTO);
    EXPECT_EQ(film.info.accumulated_samples, 6);
  } else {
    EXPECT_EQ(renderer.ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO), sparkium::RENDER_PIPELINE_RT_FALLBACK);
    if (GetParam() == sparkium::RenderBackend::CUDA)
      EXPECT_THROW(renderer.Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RAY_TRACING), std::runtime_error);
  }
}

TEST_P(ComputeBackendTest, ShaderGraphDirectLightingMatchesPrincipled) {
  sparkium::Core renderer(core.get());
  const std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}};
  const uint32_t indices[]{0, 1, 2, 0, 2, 3};
  Mesh<> mesh(4, 6, indices, positions.data());
  sparkium::GeometryMesh geometry(&renderer, mesh);
  sparkium::MaterialShaderGraph graph(&renderer, sparkium::CodeLines(R"(
GraphSurface EvaluateShaderGraph(SP_CONTEXT HitRecord hit, float3 view_direction,
    int bounce, int ray_type, bool shadow, ByteAddressBuffer data) {
  GraphSurface surface = (GraphSurface)0;
  surface.base_color = float3(0.2, 0.4, 0.6);
  surface.normal = hit.normal;
  surface.ior = 1.45;
  surface.opacity = 1;
  surface.shadow_opacity = -1;
  return surface;
}
)"),
                                      {});
  sparkium::MaterialPrincipled principled(&renderer, {0.2f, 0.4f, 0.6f});
  principled.roughness = 0;
  sparkium::EntityPointLight light(&renderer, {0, 0, 2}, {1, 1, 1}, 10);
  sparkium::Camera camera(&renderer, glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(20.0f), 1.0f);
  std::array<std::vector<glm::vec4>, 2> images;
  std::array<sparkium::Material *, 2> materials{&principled, &graph};
  for (int i = 0; i < 2; ++i) {
    sparkium::EntityGeometryMaterial entity(&renderer, &geometry, materials[i]);
    sparkium::Scene scene(&renderer);
    scene.AddEntity(&entity);
    scene.AddEntity(&light);
    scene.settings.samples_per_dispatch = 4;
    scene.settings.max_bounces = 1;
    sparkium::Film film(&renderer, 8, 8);
    renderer.Render(&scene, &camera, &film);
    images[i].resize(64);
    film.GetRawImage()->DownloadData(images[i].data());
  }
  // One bounce isolates direct-light weighting, including its finite-value guard.
  // The faulty LLVM vector guard leaves raw light power instead of BSDF / PDF.
  for (int i = 0; i < 64; ++i)
    for (int c = 0; c < 3; ++c) {
      ASSERT_GT(images[0][i][c], 0.0f);
      ASSERT_LT(images[0][i][c], 0.5f);
      EXPECT_NEAR(images[1][i][c], images[0][i][c], 1e-5f) << "pixel " << i << " channel " << c;
    }
}

TEST_P(ComputeBackendTest, MeshMaterialAndBVHWithAnalyticEmission) {
  sparkium::Core renderer(core.get());
  const std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  const uint32_t indices[]{0, 1, 2, 0, 2, 3};
  Mesh<> mesh(4, 6, indices, positions.data());
  sparkium::GeometryMesh geometry(&renderer, mesh);
  const glm::vec3 emission{0.25f, 0.5f, 1.0f};
  sparkium::MaterialLight material(&renderer, emission, true, true);
  sparkium::EntityGeometryMaterial entity(&renderer, &geometry, &material);
  sparkium::Scene scene(&renderer);
  scene.AddEntity(&entity);
  scene.settings.samples_per_dispatch = 2;
  scene.settings.max_bounces = 2;
  scene.settings.background_color = {0.01f, 0.02f, 0.03f};
  sparkium::Camera camera(&renderer, glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                          glm::radians(60.0f), 17.0f / 13.0f);
  sparkium::Film film(&renderer, 17, 13);
  renderer.Render(&scene, &camera, &film);
  std::vector<glm::vec4> pixels(17 * 13);
  film.GetRawImage()->DownloadData(pixels.data());
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(pixels[6 * 17 + 8][c], emission[c], 1e-5f);
    EXPECT_NEAR(pixels[0][c], scene.settings.background_color[c], 1e-5f);
  }
  for (const auto &pixel : pixels)
    for (int c = 0; c < 4; ++c)
      EXPECT_TRUE(std::isfinite(pixel[c]));
}

TEST_P(ComputeBackendTest, ExplicitContextRebindingAndCachedModules) {
  sparkium::Core renderer(core.get());
  auto vfs = renderer.GetShadersVFS();
  vfs.WriteFile("context_test.hlsl", R"(
#include "compute_contract.hlsli"
struct Settings { uint scale; uint bias; };
#ifdef SPARKIUM_CPU_FUNCTIONS
#define PARAMS ComputeResource<ConstantBuffer<Settings>>(compute_context,0)
#define OUTPUT ComputeResource<RWByteAddressBuffer>(compute_context,1)
#else
SP_RESOURCE(ConstantBuffer<Settings>, parameters, b0, 0);
SP_RESOURCE(RWByteAddressBuffer, output, u0, 1);
#define PARAMS parameters
#define OUTPUT output
#endif
SP_NUMTHREADS(2,3,2) void Main(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  uint index = id.x + 10*(id.y + 12*id.z);
  OUTPUT.Store(index*4,index*PARAMS.scale+PARAMS.bias);
}
)");
  std::unique_ptr<graphics::Shader> a, b;
  ASSERT_EQ(core->CreateShader(vfs, "context_test.hlsl", "Main", "cs_6_0", &a), 0);
  ASSERT_EQ(core->CreateShader(vfs, "context_test.hlsl", "Main", "cs_6_0", &b), 0);
  std::unique_ptr<graphics::Buffer> parameters, output;
  core->CreateBuffer(8, graphics::BUFFER_TYPE_STATIC, &parameters);
  core->CreateBuffer(720 * 4, graphics::BUFFER_TYPE_STATIC, &output);
  unsigned bias = 7;
  for (auto shader : {a.get(), b.get(), a.get()}) {
    std::unique_ptr<graphics::ComputeProgram> program;
    core->CreateComputeProgram(shader, &program);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    program->Finalize();
    uint32_t data[]{3, bias};
    parameters->UploadData(data, sizeof(data));
    std::unique_ptr<graphics::CommandContext> cmd;
    core->CreateCommandContext(&cmd);
    cmd->CmdBindComputeProgram(program.get());
    cmd->CmdBindResources(0, std::vector<graphics::Buffer *>{parameters.get()}, graphics::BIND_POINT_COMPUTE);
    cmd->CmdBindResources(1, std::vector<graphics::Buffer *>{output.get()}, graphics::BIND_POINT_COMPUTE);
    cmd->CmdDispatch(5, 4, 3);
    core->SubmitCommandContext(cmd.get());
    std::array<uint32_t, 720> actual;
    output->DownloadData(actual.data(), sizeof(actual));
    for (unsigned i = 0; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], i * 3 + bias);
    bias += 13;
  }
}

TEST_P(ComputeBackendTest, CpuSahTraversalMatchesIndependentTriangleOracle) {
  if (GetParam() != sparkium::RenderBackend::CPU)
    GTEST_SKIP() << "CPU-specific acceleration structure";
  using namespace sparkium::raytracing;
  constexpr uint32_t triangles = 257, ray_count = 1024;
  const uint32_t positions_offset = 52, indices_offset = positions_offset + triangles * 36;
  std::vector<uint8_t> geometry(indices_offset + triangles * 12);
  auto put = [&](size_t offset, const auto &value) { std::memcpy(geometry.data() + offset, &value, sizeof(value)); };
  put(8, positions_offset);
  put(12, uint32_t(12));
  put(48, indices_offset);
  std::mt19937 random(103);
  std::uniform_real_distribution<float> coordinate(-1, 1);
  std::vector<glm::vec3> centers;
  for (uint32_t i = 0; i < triangles; ++i) {
    glm::vec3 center{coordinate(random), coordinate(random), -float(i) * 0.003f};
    centers.push_back(center);
    for (uint32_t v = 0; v < 3; ++v) {
      glm::vec3 point = center + glm::vec3(v == 0 ? -0.2f : v == 1 ? 0.2f : 0, v == 2 ? 0.2f : -0.2f, 0);
      put(positions_offset + (i * 3 + v) * 12, point);
      put(indices_offset + (i * 3 + v) * 4, i * 3 + v);
    }
  }

  auto blas = BuildCpuMeshBvh(geometry, triangles);
  const glm::vec3 scale{-1.5f, 2.0f, 0.5f};
  CpuBounds world;
  for (unsigned i = 0; i < 8; ++i)
    world.Extend(scale * glm::vec3(i & 1 ? blas.bounds.hi.x : blas.bounds.lo.x,
                                   i & 2 ? blas.bounds.hi.y : blas.bounds.lo.y,
                                   i & 4 ? blas.bounds.hi.z : blas.bounds.lo.z));
  auto tlas = BuildCpuBvh({world});
  uint32_t root = uint32_t(tlas.bytes.size());
  tlas.bytes.insert(tlas.bytes.end(), blas.bytes.begin(), blas.bytes.end());

  struct Instance {
    glm::mat4x3 object_to_world, world_to_object;
    uint32_t root, geometry, material, count;
  };

  static_assert(sizeof(Instance) == 112);
  glm::mat4 transform = glm::scale(glm::mat4(1), scale);
  Instance instance{glm::mat4x3(transform), glm::mat4x3(glm::inverse(transform)), root, 0, 0, triangles};
  std::array<uint8_t, 128> instances{};
  uint32_t one = 1;
  std::memcpy(instances.data(), &one, 4);
  std::memcpy(instances.data() + 16, &instance, sizeof(instance));

  struct Ray {
    glm::vec3 origin;
    float t_min;
    glm::vec3 direction;
    float t_max;
  };

  struct Result {
    float distance;
    uint32_t primitive, found, any;
  };

  std::vector<Ray> rays(ray_count);
  std::vector<Result> expected(ray_count);
  for (uint32_t i = 0; i < ray_count; ++i) {
    rays[i] = {{coordinate(random) * 2, coordinate(random) * 2, 1},
               i % 3 == 0 ? 0.6f : 0.0f,
               {0, 0, -2},
               i % 5 == 0 ? 0.5f : 2.0f};
    auto &e = expected[i];
    e = {rays[i].t_max, 0xffffffff, 0, 0};
    const double x = double(rays[i].origin.x) / scale.x, y = double(rays[i].origin.y) / scale.y;
    for (uint32_t j = 0; j < triangles; ++j) {
      // Independent analytic barycentric test for the known XY triangles.
      const auto c = centers[j];
      double v = (y - (double(c.y) - 0.2)) / 0.4;
      double u = (x - (double(c.x) - 0.2) - 0.2 * v) / 0.4;
      double t = (1.0 - double(c.z) * scale.z) / 2.0;
      if (u >= 0 && v >= 0 && u + v <= 1 && t >= rays[i].t_min && t < e.distance) {
        e.distance = float(t);
        e.primitive = j;
        e.found = e.any = 1;
      }
    }
  }
  sparkium::Core renderer(core.get());
  auto vfs = renderer.GetShadersVFS();
  vfs.WriteFile("cpu_query_test.hlsl", R"(
#include "compute_contract.hlsli"
#ifdef SPARKIUM_CPU_FUNCTIONS
#define SP_BINDING_data_buffers ComputeResource<ComputeArray<RWByteAddressBuffer>>(compute_context,0)
#define SP_BINDING_software_nodes ComputeResource<RWByteAddressBuffer>(compute_context,1)
#define SP_BINDING_software_instances ComputeResource<RWByteAddressBuffer>(compute_context,2)
#define RAYS ComputeResource<RWByteAddressBuffer>(compute_context,3)
#define OUTPUT ComputeResource<RWByteAddressBuffer>(compute_context,4)
#else
SP_ARRAY_RESOURCE(ByteAddressBuffer,data_buffers,t0,0);
SP_RESOURCE(ByteAddressBuffer,software_nodes,t0,1);
SP_RESOURCE(ByteAddressBuffer,software_instances,t0,2);
SP_RESOURCE(ByteAddressBuffer,rays,t0,3);
SP_RESOURCE(RWByteAddressBuffer,output,u0,4);
#define SP_BINDING_data_buffers data_buffers
#define SP_BINDING_software_nodes software_nodes
#define SP_BINDING_software_instances software_instances
#define RAYS rays
#define OUTPUT output
#endif
#include "software/cpu_traversal.hlsli"
SP_NUMTHREADS(64,1,1) void Main(SP_CONTEXT uint3 id:SV_DispatchThreadID) {
  SP_RAY ray;float4 a=asfloat(RAYS.Load4(id.x*32)),b=asfloat(RAYS.Load4(id.x*32+16));
  ray.Origin=a.xyz;ray.TMin=a.w;ray.Direction=b.xyz;ray.TMax=b.w;
  SoftwareHit hit,occluder;bool found=InlineIntersect(SP_CONTEXT_ARG ray,false,hit);
  bool any_hit=InlineIntersect(SP_CONTEXT_ARG ray,true,occluder);
  OUTPUT.Store4(id.x*16,uint4(asuint(hit.distance),hit.primitive,uint(found),uint(any_hit)));
}
)");
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(core->CreateShader(vfs, "cpu_query_test.hlsl", "Main", "cs_6_0", &shader), 0);
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  std::array<std::unique_ptr<graphics::Buffer>, 5> buffers;
  const void *data[]{geometry.data(), tlas.bytes.data(), instances.data(), rays.data(), nullptr};
  const size_t sizes[]{geometry.size(), tlas.bytes.size(), instances.size(), rays.size() * sizeof(Ray),
                       ray_count * sizeof(Result)};
  for (unsigned i = 0; i < 5; ++i) {
    core->CreateBuffer(sizes[i], graphics::BUFFER_TYPE_STATIC, &buffers[i]);
    if (data[i])
      buffers[i]->UploadData(data[i], sizes[i]);
    program->AddResourceBinding(
        i == 4 ? graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER : graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  }
  program->Finalize();
  std::unique_ptr<graphics::CommandContext> cmd;
  core->CreateCommandContext(&cmd);
  cmd->CmdBindComputeProgram(program.get());
  for (unsigned i = 0; i < 5; ++i)
    cmd->CmdBindResources(i, std::vector<graphics::Buffer *>{buffers[i].get()}, graphics::BIND_POINT_COMPUTE);
  cmd->CmdDispatch(ray_count / 64, 1, 1);
  core->SubmitCommandContext(cmd.get());
  std::vector<Result> actual(ray_count);
  buffers[4]->DownloadData(actual.data(), actual.size() * sizeof(Result));
  for (unsigned i = 0; i < ray_count; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].found, expected[i].found);
    EXPECT_EQ(actual[i].any, expected[i].any);
    if (expected[i].found) {
      EXPECT_EQ(actual[i].primitive, expected[i].primitive);
      EXPECT_NEAR(actual[i].distance, expected[i].distance, 1e-6f);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Compute,
                         ComputeBackendTest,
                         testing::Values(sparkium::RenderBackend::CPU, sparkium::RenderBackend::CUDA),
                         [](const testing::TestParamInfo<sparkium::RenderBackend> &p) {
                           return sparkium::BackendName(p.param);
                         });
}  // namespace
