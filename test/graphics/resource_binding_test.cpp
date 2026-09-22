#include <gtest/gtest.h>
#include <long_march.h>

#include <numeric>
#include <stdexcept>

#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/graphics/backend/metal/metal_program.h"
#endif

using namespace grassland;
using namespace grassland::graphics;

TEST(ResourceBinding, AllResourceKindsCompileForDXILAndSPIRV) {
  const std::vector<std::pair<ResourceType, int>> layout = {{RESOURCE_TYPE_UNIFORM_BUFFER, 1},
                                                            {RESOURCE_TYPE_STORAGE_BUFFER, 2},
                                                            {RESOURCE_TYPE_IMAGE, 2},
                                                            {RESOURCE_TYPE_WRITABLE_IMAGE, 1},
                                                            {RESOURCE_TYPE_SAMPLER, 1},
                                                            {RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1},
                                                            {RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1}};
  const std::string body = R"(
struct Params { uint index; };
RESOURCE_BINDING(0, ConstantBuffer<Params>, params);
RESOURCE_BINDING(1, StructuredBuffer<float4>, inputs);
RESOURCE_BINDING(2, Texture2D<float4>, images);
RESOURCE_BINDING(3, RWTexture2D<float4>, result);
RESOURCE_BINDING(4, SamplerState, texture_sampler);
RESOURCE_BINDING(5, RaytracingAccelerationStructure, scene);
RESOURCE_BINDING(6, RWStructuredBuffer<float4>, output);
[numthreads(1,1,1)] void Main() {
  RayDesc ray = {float3(0,0,0), 0, float3(0,0,1), 100};
  RayQuery<RAY_FLAG_NONE> query;
  query.TraceRayInline(scene, 0, 255, ray);
  while (query.Proceed()) {}
  float4 value = inputs[params.index][0] + images[params.index].SampleLevel(texture_sampler, .5, 0);
  output[0] = value + query.CommittedStatus();
  result[uint2(0,0)] = value;
})";
  for (auto api : {BACKEND_API_METAL, BACKEND_API_VULKAN, BACKEND_API_D3D12}) {
    SCOPED_TRACE(int(api));
    std::vector<std::string> args;
    if (api != BACKEND_API_D3D12)
      args = {"-spirv", "-fspv-target-env=vulkan1.2", "-fvk-use-dx-layout"};
    auto blob = CompileShader(ShaderCode::ResourceBindingDefinitions(api, layout) + body, "Main", "cs_6_5", args);
    EXPECT_FALSE(blob.data.empty());
  }
  EXPECT_THROW(ShaderCode::ResourceBindingDefinitions(BACKEND_API_METAL, {{RESOURCE_TYPE_IMAGE, 0}}),
               std::invalid_argument);
}

TEST(ResourceBinding, RayTracingLibraryEntryPointsCompile) {
  const std::string source = R"(
RESOURCE_BINDING(0, RWStructuredBuffer<float>, output);
struct Payload { float value; };
[shader("raygeneration")] void RayGen() { output[0] = 1; }
[shader("miss")] void Miss(inout Payload p) { p.value = 2; }
[shader("closesthit")] void Hit(inout Payload p, BuiltInTriangleIntersectionAttributes attr) { p.value = attr.barycentrics.x; }
[shader("anyhit")] void AnyHit(inout Payload p, BuiltInTriangleIntersectionAttributes attr) { if (p.value < attr.barycentrics.x) IgnoreHit(); }
[shader("intersection")] void Intersection() { BuiltInTriangleIntersectionAttributes attr = {float2(0,0)}; ReportHit(1, 0, attr); }
[shader("callable")] void Callable(inout Payload p) { p.value = output[0]; }
)";
  for (auto api : {BACKEND_API_VULKAN, BACKEND_API_D3D12}) {
    std::vector<std::string> args;
    if (api == BACKEND_API_VULKAN)
      args = {"-spirv", "-fspv-target-env=vulkan1.2"};
    auto generated = ShaderCode::ResourceBindingDefinitions(api, {{RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1}}) + source;
    for (const char *entry : {"RayGen", "Miss", "Hit", "AnyHit", "Intersection", "Callable"}) {
      SCOPED_TRACE(testing::Message() << int(api) << " " << entry);
      EXPECT_FALSE(CompileShader(generated, entry, "lib_6_3", args).data.empty());
    }
  }
}

class ResourceBindingGPU : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    ASSERT_EQ(CreateCore(GetParam(), {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  }

  std::unique_ptr<Core> core;
};

std::vector<BackendAPI> TestBackends() {
  std::vector<BackendAPI> backends;
#if defined(LONGMARCH_METAL_ENABLED)
  backends.push_back(BACKEND_API_METAL);
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  backends.push_back(BACKEND_API_VULKAN);
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  backends.push_back(BACKEND_API_D3D12);
#endif
  return backends;
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, ResourceBindingGPU, testing::ValuesIn(TestBackends()));

TEST_P(ResourceBindingGPU, RayTracingProgramAcceptsShaderCode) {
  if (!core->DeviceRayTracingSupport())
    GTEST_SKIP() << "native ray tracing pipeline unavailable";
  ShaderCode raygen(
      "RESOURCE_BINDING(0, RWStructuredBuffer<float>, output); [shader(\"raygeneration\")] void Main() { output[0] = 7; }",
      "Main", "lib_6_3");
  ShaderCode miss("struct Payload { float value; }; [shader(\"miss\")] void Main(inout Payload p) { p.value = 2; }",
                  "Main", "lib_6_3");
  ShaderCode hit(
      "struct Payload { float value; }; [shader(\"closesthit\")] void Main(inout Payload p, BuiltInTriangleIntersectionAttributes a) { p.value = 3; }",
      "Main", "lib_6_3");
  std::unique_ptr<RayTracingProgram> program;
  core->CreateRayTracingProgram(&program);
  program->AddRayGenShader(raygen);
  program->AddMissShader(miss);
  program->AddHitGroup(hit);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::unique_ptr<Buffer> output;
  core->CreateBuffer(4, BUFFER_TYPE_STATIC, &output);
  std::unique_ptr<CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdBindRayTracingProgram(program.get());
  commands->CmdBindResources(0, {output.get()}, BIND_POINT_RAYTRACING);
  commands->CmdDispatchRays(1, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  float value;
  output->DownloadData(&value, 4);
  EXPECT_FLOAT_EQ(value, 7);
}

TEST_P(ResourceBindingGPU, TexturesSamplersConstantsAndLegacyCompute) {
  ShaderCode code(R"(
struct Params { float4 multiplier; };
RESOURCE_BINDING(0, ConstantBuffer<Params>, params);
RESOURCE_BINDING(1, Texture2D<float4>, images);
RESOURCE_BINDING(2, SamplerState, texture_sampler);
RESOURCE_BINDING(3, RWTexture2D<float4>, output);
[numthreads(1,1,1)] void Main() {
  output[uint2(0,0)] = images[0].SampleLevel(texture_sampler, float2(.5,.5), 0) * params.multiplier;
})",
                  "Main", "cs_6_0");
  std::unique_ptr<ComputeProgram> program;
  core->CreateComputeProgram(code, &program);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_IMAGE, 2);
  program->AddResourceBinding(RESOURCE_TYPE_SAMPLER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program->Finalize();
  std::unique_ptr<Buffer> params;
  core->CreateBuffer(256, BUFFER_TYPE_STATIC, &params);
  float values[] = {2, 3, 4, 1};
  params->UploadData(values, sizeof(values));
  std::unique_ptr<Image> input, output;
  core->CreateImage(1, 1, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &input);
  core->CreateImage(1, 1, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output);
  input->UploadData(values);
  std::unique_ptr<Sampler> sampler;
  core->CreateSampler({}, &sampler);
  std::unique_ptr<CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, {params.get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {input.get(), input.get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(2, {sampler.get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(3, {output.get()}, BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  output->DownloadData(values);
  EXPECT_FLOAT_EQ(values[0], 4);
  EXPECT_FLOAT_EQ(values[1], 9);
  EXPECT_FLOAT_EQ(values[2], 16);
  EXPECT_FLOAT_EQ(values[3], 1);

  std::unique_ptr<Shader> legacy;
  ASSERT_EQ(
      core->CreateShader(
          "RWTexture2D<float4> output : register(u0, space0); [numthreads(1,1,1)] void Main() { output[uint2(0,0)] = float4(1,2,3,4); }",
          "Main", "cs_6_0", &legacy),
      0);
  core->CreateComputeProgram(legacy.get(), &program);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program->Finalize();
  core->CreateCommandContext(&commands);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, {output.get()}, BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  output->DownloadData(values);
  EXPECT_FLOAT_EQ(values[0], 1);
  EXPECT_FLOAT_EQ(values[3], 4);
}

TEST_P(ResourceBindingGPU, GraphicsStagesCanMixSourceAndLegacyShaders) {
  std::unique_ptr<Shader> legacy_vertex;
  ASSERT_EQ(core->CreateShader(R"(
struct Params { float4 value; };
ConstantBuffer<Params> params : register(b0, space0);
float4 Main(uint id : SV_VertexID) : SV_Position {
  return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, params.value.x, 1);
})",
                               "Main", "vs_6_0", &legacy_vertex),
            0);
  ShaderCode pixel(R"(
struct Params { float4 value; };
RESOURCE_BINDING(1, ConstantBuffer<Params>, params);
float4 Main() : SV_Target { return params.value; }
)",
                   "Main", "ps_6_0");
  std::unique_ptr<Program> program;
  core->CreateProgram({IMAGE_FORMAT_R32G32B32A32_SFLOAT}, IMAGE_FORMAT_UNDEFINED, &program);
  program->BindShader(legacy_vertex.get(), SHADER_TYPE_VERTEX);
  program->BindShader(pixel, SHADER_TYPE_PIXEL);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->SetCullMode(CULL_MODE_NONE);
  program->Finalize();
  std::unique_ptr<Buffer> vertex_params, pixel_params;
  core->CreateBuffer(256, BUFFER_TYPE_STATIC, &vertex_params);
  core->CreateBuffer(256, BUFFER_TYPE_STATIC, &pixel_params);
  float values[] = {0, 0, 0, 0};
  vertex_params->UploadData(values, sizeof(values));
  const float color[] = {.25f, .5f, .75f, 1};
  pixel_params->UploadData(color, sizeof(color));
  std::unique_ptr<Image> output;
  core->CreateImage(4, 4, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output);
  std::unique_ptr<CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdBindProgram(program.get());
  commands->CmdBindResources(0, {vertex_params.get()}, BIND_POINT_GRAPHICS);
  commands->CmdBindResources(1, {pixel_params.get()}, BIND_POINT_GRAPHICS);
  commands->CmdBeginRendering({output.get()}, nullptr);
  commands->CmdSetViewport({0, 0, 4, 4, 0, 1});
  commands->CmdSetScissor({0, 0, 4, 4});
  commands->CmdSetPrimitiveTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  commands->CmdDraw(3, 1, 0, 0);
  commands->CmdEndRendering();
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  float image[64];
  output->DownloadData(image);
  for (int i = 0; i < 64; ++i)
    EXPECT_FLOAT_EQ(image[i], color[i % 4]);
}

#if defined(LONGMARCH_METAL_ENABLED)
class ResourceBindingMetal : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(CreateCore(BACKEND_API_METAL, {}, &core), 0);
    ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  }

  std::unique_ptr<Core> core;
};

TEST_F(ResourceBindingMetal, ManySlotsArrayOffsetsSnapshotsAndSourceOwnership) {
  constexpr int slots = 24;
  std::string declarations, expression;
  std::vector<std::unique_ptr<Buffer>> buffers(slots + 1);
  for (int i = 0; i < slots; ++i) {
    declarations +=
        "RESOURCE_BINDING(" + std::to_string(i) + ", StructuredBuffer<uint>, input" + std::to_string(i) + ");\n";
    expression +=
        (i ? " + " : "") + std::string("input") + std::to_string(i) + (i == 0 ? "[0][0] + input0[1][0]" : "[0]");
  }
  declarations += "RESOURCE_BINDING(24, RWStructuredBuffer<uint>, output);\n";
  std::unique_ptr<ComputeProgram> program;
  {
    VirtualFileSystem vfs;
    vfs.WriteFile("resources.hlsli", declarations);
    vfs.WriteFile("main.hlsl",
                  "#include \"resources.hlsli\"\n[numthreads(1,1,1)] void Main() { output[0] = " + expression + "; }");
    ShaderCode code(vfs, "main.hlsl", "Main", "cs_6_0");
    ASSERT_EQ(core->CreateComputeProgram(code, &program), 0);
    vfs.WriteFile("resources.hlsli", "invalid original source after snapshot");
  }
  for (int i = 0; i <= slots; ++i) {
    core->CreateBuffer(4, BUFFER_TYPE_STATIC, &buffers[i]);
    uint32_t value = i + 1;
    buffers[i]->UploadData(&value, 4);
  }
  for (int i = 0; i < slots; ++i)
    program->AddResourceBinding(RESOURCE_TYPE_STORAGE_BUFFER, i == 0 ? 2 : 1);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  auto metal = dynamic_cast<backend::MetalComputeProgram *>(program.get());
  ASSERT_TRUE(metal);
  EXPECT_EQ(metal->stage.arguments.size(), 1);
  EXPECT_TRUE(metal->stage.packed);
  EXPECT_EQ(metal->stage.resource_indices.at(24), 25);
  EXPECT_THROW(program->AddResourceBinding(RESOURCE_TYPE_IMAGE, 1), std::logic_error);
  std::unique_ptr<Buffer> output1, output2;
  core->CreateBuffer(4, BUFFER_TYPE_STATIC, &output1);
  core->CreateBuffer(4, BUFFER_TYPE_STATIC, &output2);
  std::unique_ptr<CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, {buffers[0].get(), buffers[24].get()}, BIND_POINT_COMPUTE);
  for (int i = 1; i < slots; ++i)
    commands->CmdBindResources(i, {buffers[i].get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(24, {output1.get()}, BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  commands->CmdBindResources(0, {buffers[0].get(), buffers[0].get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(24, {output2.get()}, BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  commands.reset();
  uint32_t result;
  output1->DownloadData(&result, 4);
  EXPECT_EQ(result, 325);
  output2->DownloadData(&result, 4);
  EXPECT_EQ(result, 301);
}

TEST_F(ResourceBindingMetal, CompilationIsDeferredAndErrorsAreReported) {
  std::unique_ptr<Shader> legacy;
  core->CreateShader("[numthreads(1,1,1)] void Main() {}", "Main", "cs_6_0", &legacy);
  std::unique_ptr<ComputeProgram> oversized;
  core->CreateComputeProgram(legacy.get(), &oversized);
  for (int i = 0; i < 9; ++i)
    oversized->AddResourceBinding(RESOURCE_TYPE_STORAGE_BUFFER, 1);
  EXPECT_THROW(oversized->Finalize(), std::runtime_error);
  std::unique_ptr<ComputeProgram> program;
  EXPECT_EQ(core->CreateComputeProgram(ShaderCode("syntax error", "Main", "cs_6_0"), &program), 0);
  EXPECT_THROW(program->Finalize(), std::runtime_error);
  VirtualFileSystem empty;
  EXPECT_EQ(core->CreateComputeProgram(ShaderCode(empty, "missing.hlsl", "Main", "cs_6_0"), &program), 0);
  EXPECT_THROW(program->Finalize(), std::runtime_error);
  EXPECT_EQ(
      core->CreateComputeProgram(
          ShaderCode(
              "RESOURCE_BINDING(2, RWByteAddressBuffer, output);\n[numthreads(1,1,1)] void Main() { output.Store(0, 1); }",
              "Main", "cs_6_0"),
          &program),
      0);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  EXPECT_THROW(program->Finalize(), std::runtime_error);
}

TEST_F(ResourceBindingMetal, OneSourceCompilesIndependentlyForEachProgramLayout) {
  ShaderCode code(R"(
RESOURCE_BINDING(0, ByteAddressBuffer, inputs);
RESOURCE_BINDING(1, RWStructuredBuffer<uint>, output);
[numthreads(1,1,1)] void Main() { output[0] = inputs[0].Load(0) + inputs[1].Load(0) + EXTRA; }
)",
                  "Main", "cs_6_0", {"-DEXTRA=7"});
  std::unique_ptr<Buffer> input, output;
  core->CreateBuffer(4, BUFFER_TYPE_STATIC, &input);
  core->CreateBuffer(4, BUFFER_TYPE_STATIC, &output);
  uint32_t value = 10;
  input->UploadData(&value, 4);
  std::vector<std::unique_ptr<ComputeProgram>> programs(2);
  for (int i = 0; i < 2; ++i) {
    core->CreateComputeProgram(code, &programs[i]);
    programs[i]->AddResourceBinding(RESOURCE_TYPE_STORAGE_BUFFER, i + 2);
    programs[i]->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
    programs[i]->Finalize();
    auto metal = dynamic_cast<backend::MetalComputeProgram *>(programs[i].get());
    EXPECT_EQ(metal->stage.resource_indices.at(1), i + 2);
  }
  for (int i : {0, 1, 0}) {
    std::unique_ptr<CommandContext> commands;
    core->CreateCommandContext(&commands);
    commands->CmdBindComputeProgram(programs[i].get());
    commands->CmdBindResources(0, std::vector<Buffer *>(i + 2, input.get()), BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {output.get()}, BIND_POINT_COMPUTE);
    commands->CmdDispatch(1, 1, 1);
    ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
    output->DownloadData(&value, 4);
    EXPECT_EQ(value, 27);
  }
}

TEST_F(ResourceBindingMetal, PackedAccelerationStructureUsesNonzeroArgumentIndex) {
  if (!core->DeviceRayQuerySupport())
    GTEST_SKIP() << "ray query unavailable";
  std::unique_ptr<Buffer> vertices, indices, output;
  core->CreateBuffer(36, BUFFER_TYPE_STATIC, &vertices);
  core->CreateBuffer(12, BUFFER_TYPE_STATIC, &indices);
  core->CreateBuffer(4, BUFFER_TYPE_STATIC, &output);
  const float points[] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
  const uint32_t faces[] = {0, 1, 2};
  vertices->UploadData(points, sizeof(points));
  indices->UploadData(faces, sizeof(faces));
  std::unique_ptr<AccelerationStructure> triangle, scene;
  core->CreateBottomLevelAccelerationStructure(vertices.get(), indices.get(), 12, &triangle);
  core->CreateTopLevelAccelerationStructure({triangle->MakeInstance(glm::mat4(1.0f))}, &scene);
  ShaderCode code(R"(
RESOURCE_BINDING(0, RWStructuredBuffer<float>, output);
RESOURCE_BINDING(1, RaytracingAccelerationStructure, scene);
[numthreads(1,1,1)] void Main() {
  RayDesc ray = {float3(0,0,-2), 0, float3(0,0,1), 100};
  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(scene, 0, 255, ray);
  while (query.Proceed()) {}
  output[0] = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : -1;
})",
                  "Main", "cs_6_5");
  std::unique_ptr<ComputeProgram> program;
  core->CreateComputeProgram(code, &program);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
  program->Finalize();
  std::unique_ptr<CommandContext> commands;
  core->CreateCommandContext(&commands);
  commands->CmdBindComputeProgram(program.get());
  commands->CmdBindResources(0, {output.get()}, BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, scene.get(), BIND_POINT_COMPUTE);
  commands->CmdDispatch(1, 1, 1);
  ASSERT_EQ(core->SubmitCommandContext(commands.get()), 0);
  float distance;
  output->DownloadData(&distance, 4);
  EXPECT_FLOAT_EQ(distance, 2);
}
#endif
