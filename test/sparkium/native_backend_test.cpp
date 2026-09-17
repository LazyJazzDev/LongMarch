#include <gtest/gtest.h>
#include <long_march.h>

#include <array>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>

using namespace grassland;
namespace {
class NativeBackendTest : public testing::TestWithParam<graphics::BackendAPI> {
 protected:
  void SetUp() override {
    if (!graphics::SupportBackendAPI(GetParam()))
      GTEST_SKIP() << "backend not compiled";
    ASSERT_EQ(graphics::CreateCore(GetParam(), {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
    EXPECT_FALSE(core->DeviceRayTracingSupport());
    EXPECT_FALSE(core->DeviceRayQuerySupport());
    EXPECT_EQ(core->API(), GetParam());
  }
  std::unique_ptr<graphics::Core> core;
};

TEST_P(NativeBackendTest, DescriptorArraysRangesAndIndependentEntryPoints) {
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

TEST_P(NativeBackendTest, TextureArraysFilteringAndPartialImageTransfers) {
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

TEST_P(NativeBackendTest, HDRFloat3AndNearestAddressModes) {
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

TEST_P(NativeBackendTest, FrameAccumulationResetAndBackground) {
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
}

INSTANTIATE_TEST_SUITE_P(Native,
                         NativeBackendTest,
                         testing::Values(graphics::BACKEND_API_CPU, graphics::BACKEND_API_CUDA),
                         [](const testing::TestParamInfo<graphics::BackendAPI> &p) {
                           return graphics::BackendAPIString(p.param);
                         });
}  // namespace
