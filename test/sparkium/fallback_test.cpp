#include <gtest/gtest.h>
#include <long_march.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <numeric>
#include <random>
#include <tuple>

#include "../../demo/sparkium_backend.h"
#include "grassland/graphics/backend/backend.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "sparkium/core/frame_profile.h"
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#include "sparkium/pipelines/raytracing/core/camera.h"
#include "sparkium/pipelines/raytracing/core/core.h"
#include "sparkium/pipelines/raytracing/core/software_pipeline.h"
#include "sparkium/pipelines/raytracing/entity/entities.h"
#include "sparkium/pipelines/raytracing/geometry/geometry_mesh.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

using namespace grassland;
namespace vulkan = grassland::graphics::backend::vulkan;

namespace {
struct Ray {
  glm::vec3 origin;
  float t_min;
  glm::vec3 direction;
  float t_max;
};

struct Hit {
  float distance;
  uint32_t instance, primitive, found;
  float u, v, padding[2];
};

static_assert(sizeof(Ray) == 32 && sizeof(Hit) == 32);

class SoftwareBVHTest : public testing::Test {
 protected:
  void SetUp() override {
    const char *backend = std::getenv("SPARKIUM_TEST_BACKEND");
    const bool debug = std::getenv("SPARKIUM_TEST_DEBUG") != nullptr;
    api = backend ? ParseSparkiumBackend(backend) : rhi::BACKEND_API_DEFAULT;
    native_core = rhi::CreateCore(api, {debug});
    ASSERT_TRUE(native_core);
    device = native_core->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device);
    core = std::make_unique<sparkium::Core>(device.Get());
  }

  rhi::BackendAPI api{};
  rhi::ObjPtr<rhi::ICore> native_core;
  rhi::ObjPtr<rhi::IDevice> device;
  std::unique_ptr<sparkium::Core> core;
};

TEST_F(SoftwareBVHTest, ShadowTerminatorOffsetPreservesPlanesAndScalesWithGeometry) {
  auto vfs = core->GetShadersVFS();
  vfs.WriteFile("terminator_test.slang", R"(
#include "shadow_terminator.slang"
RHI_RW_STRUCTURED_BUFFER(0, 0, float4, output);
[numthreads(1,1,1)] void Main(uint3 id : SV_DispatchThreadID) {
  float3 V[3] = {float3(0,0,0), float3(1,0,0), float3(0,1,0)};
  float3 N[3] = {float3(0,0,1), float3(0.6,0,0.8), float3(0,0.6,0.8)};
  float3 bary = float3(1.0/3.0);
  output[0] = float4(ShadowTerminatorSurfaceOffset(V, N, bary, false), ShadowTerminatorWeight(0.5, 0.2, 0.1));
  output[1] = float4(ShadowTerminatorSurfaceOffset(V, N, bary, true), ShadowTerminatorWeight(0.2, 0.05, 0.1));
  output[2] = float4(ShadowTerminatorSurfaceOffset(V, N, float3(1,0,0), false), ShadowTerminatorWeight(0.05, 0.05, 0.1));
  for (int i = 0; i < 3; ++i) V[i] *= 2.0;
  output[3] = float4(ShadowTerminatorSurfaceOffset(V, N, bary, false), ShadowTerminatorWeight(0, -0.1, 0));
  for (int j = 0; j < 3; ++j) N[j] = float3(0,0,1);
  output[4] = float4(ShadowTerminatorSurfaceOffset(V, N, bary, false), ShadowTerminatorWeight(0, -0.1, 0.1));
})");
  rhi::ObjPtr<rhi::IShader> shader;
  shader = device->CreateShader("terminator_test.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &vfs, {"."});
  ASSERT_TRUE(shader);
  rhi::ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, rhi::BINDING_RW_STRUCTURED_BUFFER, 1, rhi::SHADER_VISIBILITY_ALL, 16}}}};
  rhi::ObjPtr<rhi::IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  rhi::ObjPtr<rhi::IBindingGroup> bindings = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(bindings);
  std::array<glm::vec4, 5> values;
  rhi::ObjPtr<rhi::IBuffer> output;
  output = device->CreateBuffer({sizeof(values)});
  rhi::ObjPtr<rhi::ICommandContext> commands;
  commands = device->CreateCommandContext();
  commands->CmdBindComputeProgram(program.Get());
  bindings->BindResource(0, output.Get());
  commands->CmdBindGroup(0, bindings.Get());
  commands->CmdDispatch(1, 1, 1);
  ASSERT_TRUE(sparkium::Execute(device.Get(), commands.Get()));
  EXPECT_TRUE(sparkium::DownloadBuffer(device.Get(), output.Get(), values.data(), sizeof(values)));
  // Golden values from Cycles' parabolic height and local-envelope equations.
  const std::array<glm::vec4, 5> expected{{{0.026666667f, 0.026666667f, 0.115555556f, 0},
                                           {0, 0, 0, 0.5f},
                                           {0, 0, 0, 1},
                                           {0.053333333f, 0.053333333f, 0.231111111f, 0},
                                           {0, 0, 0, 1}}};
  for (size_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(i);
    for (int axis = 0; axis < 4; ++axis)
      EXPECT_NEAR(values[i][axis], expected[i][axis], 1e-6f);
  }
}

TEST_F(SoftwareBVHTest, GeometryOffsetSettingValidatesAndUpdatesExistingBuffer) {
  sparkium::GeometryMesh mesh(core.get(), Mesh<>::Sphere(8, 8));
  auto *buffer = mesh.GetBuffer();
  EXPECT_FLOAT_EQ(mesh.GetShadowTerminatorGeometryOffset(), 0.1f);
  for (float value : {0.0f, 0.5f, 1.0f}) {
    mesh.SetShadowTerminatorGeometryOffset(value);
    EXPECT_EQ(mesh.GetBuffer(), buffer);
    EXPECT_FLOAT_EQ(mesh.GetShadowTerminatorGeometryOffset(), value);
    sparkium::GeometryMesh::Header header;
    EXPECT_TRUE(sparkium::DownloadBuffer(device.Get(), buffer, &header, sizeof(header)));
    EXPECT_FLOAT_EQ(header.shadow_terminator_geometry_offset, value);
  }
  EXPECT_THROW(mesh.SetShadowTerminatorGeometryOffset(-0.1f), std::invalid_argument);
  EXPECT_THROW(mesh.SetShadowTerminatorGeometryOffset(1.1f), std::invalid_argument);
  EXPECT_THROW(mesh.SetShadowTerminatorGeometryOffset(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW(mesh.SetShadowTerminatorGeometryOffset(std::numeric_limits<float>::infinity()), std::invalid_argument);
  EXPECT_FLOAT_EQ(mesh.GetShadowTerminatorGeometryOffset(), 1.0f);
}

TEST_F(SoftwareBVHTest, LightSelectionPartialWorkgroupDoesNotOverwriteTail) {
  rhi::ObjPtr<rhi::IShader> shader;
  shader = device->CreateShader("gather_light_power.slang", "GatherLightPowerKernel", rhi::SHADER_STAGE_COMPUTE,
                                &core->GetShadersVFS(), {"."});
  ASSERT_TRUE(shader);
  rhi::ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {{{{0, rhi::BINDING_BYTE_ADDRESS_BUFFER},
                                   {1, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER},
                                   {2, rhi::BINDING_BYTE_ADDRESS_BUFFER}}}};
  rhi::ObjPtr<rhi::IComputeProgram> program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  rhi::ObjPtr<rhi::IBindingGroup> bindings = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(bindings);
  for (uint32_t count : {1u, 5u, 63u, 65u}) {
    SCOPED_TRACE(count);
    rhi::ObjPtr<rhi::IBuffer> metadata, power, selector;
    metadata = device->CreateBuffer({count * 16});
    power = device->CreateBuffer({4});
    selector = device->CreateBuffer({(count + 65) * 4});
    std::vector<uint32_t> records(count * 4, 0), actual(count + 65, 0xdeadbeef);
    EXPECT_TRUE(sparkium::UploadBuffer(device.Get(), metadata.Get(), records.data(), records.size() * 4));
    float one = 1;
    EXPECT_TRUE(sparkium::UploadBuffer(device.Get(), power.Get(), &one, 4));
    actual[0] = count;
    EXPECT_TRUE(sparkium::UploadBuffer(device.Get(), selector.Get(), actual.data(), actual.size() * 4));
    rhi::ObjPtr<rhi::ICommandContext> commands;
    commands = device->CreateCommandContext();
    commands->CmdBindComputeProgram(program.Get());
    bindings->BindResource(0, metadata.Get());
    bindings->BindResource(1, selector.Get());
    bindings->BindResource(2, power.Get());
    commands->CmdBindGroup(0, bindings.Get());
    commands->CmdDispatch((count + 63) / 64, 1, 1);
    EXPECT_TRUE(sparkium::Execute(device.Get(), commands.Get()));
    EXPECT_TRUE(sparkium::DownloadBuffer(device.Get(), selector.Get(), actual.data(), actual.size() * 4));
    EXPECT_EQ(actual[0], count);
    for (uint32_t i = 0; i < count; ++i) {
      float value;
      std::memcpy(&value, &actual[i + 1], 4);
      EXPECT_FLOAT_EQ(value, float(i % 64 + 1));
    }
    for (size_t i = count + 1; i < actual.size(); ++i)
      EXPECT_EQ(actual[i], 0xdeadbeef);
  }
}

TEST_F(SoftwareBVHTest, JsonCameraTypesPreserveLegacyScenes) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("sparkium-camera-" + std::to_string(reinterpret_cast<uintptr_t>(core.get())) + ".json");

  struct RemoveFile {
    std::filesystem::path path;

    ~RemoveFile() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};

  auto load = [&](const std::string &parameters, std::string *error) {
    std::ofstream file(path);
    file << R"({"format":"sparkium-scene","version":1,"renderer":{},
"film":{"width":8,"height":8},"materials":{},"geometries":{},"entities":[],
"camera":{"eye":[0,0,4],"target":[0,0,0])"
         << parameters << "}}";
    file.close();
    return sparkium::JsonScene::Load(core.get(), path, error);
  };
  std::string error;
  auto pinhole = load("", &error);
  ASSERT_NE(pinhole, nullptr) << error;
  EXPECT_NE(dynamic_cast<sparkium::CameraPinhole *>(pinhole->GetCamera()), nullptr);
  auto legacy = load(R"(,"aperture_radius":0.2,"aperture_blades":6)", &error);
  ASSERT_NE(legacy, nullptr) << error;
  auto *lens = dynamic_cast<sparkium::CameraThinLens *>(legacy->GetCamera());
  ASSERT_NE(lens, nullptr);
  EXPECT_FLOAT_EQ(lens->aperture_radius, 0.2f);
  EXPECT_FLOAT_EQ(lens->focus_distance, 4.0f);
  EXPECT_EQ(lens->aperture_blades, 6);
  auto explicit_lens = load(R"(,"type":"thin_lens","aperture_radius":0)", &error);
  ASSERT_NE(explicit_lens, nullptr) << error;
  EXPECT_NE(dynamic_cast<sparkium::CameraThinLens *>(explicit_lens->GetCamera()), nullptr);
  auto explicit_pinhole = load(R"(,"type":"pinhole","aperture_radius":0.2)", &error);
  ASSERT_NE(explicit_pinhole, nullptr) << error;
  EXPECT_NE(dynamic_cast<sparkium::CameraPinhole *>(explicit_pinhole->GetCamera()), nullptr);
  EXPECT_EQ(load(R"(,"type":"unknown")", &error), nullptr);
  EXPECT_NE(error.find("unknown camera type"), std::string::npos);
}

TEST_F(SoftwareBVHTest, CameraModelsGeneratePinholeAndFocusedRays) {
  sparkium::CameraPinhole pinhole(core.get(), glm::mat4(1), glm::radians(60.0f), 1.5f);
  sparkium::CameraThinLens lens(core.get(), pinhole.view, pinhole.fovy, pinhole.aspect);
  auto sample = [&](sparkium::Camera &camera) {
    auto *adapter = sparkium::raytracing::DedicatedCast(&camera);
    auto vfs = core->GetShadersVFS();
    vfs.WriteFile("bindings.slang", R"(
#pragma once
#include "common.slang"
#include "buffer_helper.slang"
RHI_BYTE_ADDRESS_BUFFER(0, 0, camera_data);
)");
    vfs.WriteFile("camera_test.slang", "#define SPARKIUM_SOFTWARE_RT\n#include \"" + adapter->ShaderFile() +
                                           "\"\nRHI_RW_BYTE_ADDRESS_BUFFER(0, 1, output);\n"
                                           "[numthreads(1,1,1)] void Main(uint3 id : SV_DispatchThreadID) {\n"
                                           "RayGenPayload p; p.uv = float2(0.3f, -0.2f); "
                                           "p.lens_sample = float2(0.13f + 0.2f * id.x, 0.64f);\n" +
                                           adapter->EntryPoint() +
                                           "(p); output.Store4(id.x * 32, asuint(float4(p.origin, 1))); "
                                           "output.Store4(id.x * 32 + 16, asuint(float4(p.direction, 0))); }\n");
    rhi::ObjPtr<rhi::IShader> shader;
    shader = device->CreateShader("camera_test.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &vfs, {"."});
    EXPECT_TRUE(shader);
    rhi::ObjPtr<rhi::IComputeProgram> program;
    rhi::ComputeProgramDesc desc;
    desc.compute_shader = shader.Get();
    desc.pipeline_layout.groups = {{{{0, rhi::BINDING_BYTE_ADDRESS_BUFFER}, {1, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER}}}};
    program = device->CreateComputeProgram(desc);
    EXPECT_TRUE(program);
    rhi::ObjPtr<rhi::IBindingGroup> bindings = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
    rhi::ObjPtr<rhi::IBuffer> output;
    (output = device->CreateBuffer({128}));
    rhi::ObjPtr<rhi::ICommandContext> commands;
    commands = device->CreateCommandContext();
    commands->CmdBindComputeProgram(program.Get());
    bindings->BindResource(0, adapter->Buffer());
    bindings->BindResource(1, output.Get());
    commands->CmdBindGroup(0, bindings.Get());
    commands->CmdDispatch(4, 1, 1);
    EXPECT_TRUE(sparkium::Execute(device.Get(), commands.Get()));
    std::array<glm::vec4, 8> result{};
    sparkium::DownloadBuffer(device.Get(), output.Get(), result.data(), sizeof(result));
    return result;
  };
  const auto reference = sample(pinhole);
  const auto zero_aperture = sample(lens);
  const glm::vec3 expected =
      glm::normalize(glm::vec3(0.3f * 1.5f * tan(glm::radians(30.0f)), -0.2f * tan(glm::radians(30.0f)), -1.0f));
  for (size_t i = 0; i < reference.size(); ++i)
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(reference[i][c], zero_aperture[i][c], 1e-6f);
      EXPECT_NEAR(reference[i][c], i % 2 ? expected[c] : 0.0f, 1e-6f);
    }
  lens.aperture_radius = 0.5f;
  lens.focus_distance = 4.0f;
  lens.aperture_ratio = 1.7f;
  lens.aperture_rotation = 0.4f;
  for (int blades : {0, 6}) {
    lens.aperture_blades = blades;
    const auto rays = sample(lens);
    bool distinct_origin = false;
    for (int i = 0; i < 4; ++i) {
      const glm::vec3 origin(rays[i * 2]), direction(rays[i * 2 + 1]);
      distinct_origin |= glm::length(origin) > 0.1f;
      EXPECT_NEAR(origin.z, 0.0f, 1e-6f);
      EXPECT_LE(glm::length(glm::vec2(origin.x / lens.aperture_ratio, origin.y)), lens.aperture_radius + 1e-6f);
      const auto focus = origin + direction * (lens.focus_distance / -direction.z);
      const auto expected_focus = expected * (lens.focus_distance / -expected.z);
      for (int c = 0; c < 3; ++c)
        EXPECT_NEAR(focus[c], expected_focus[c], 1e-5f);
    }
    EXPECT_TRUE(distinct_origin);
  }
  // Public pose/optics changes must update the GPU data on the next access.
  pinhole.view = glm::translate(glm::mat4(1), glm::vec3(-2, -3, -4));
  const auto moved = sample(pinhole);
  for (int c = 0; c < 3; ++c)
    EXPECT_NEAR(moved[0][c], float(c + 2), 1e-6f);
}

TEST_F(SoftwareBVHTest, CameraModelsSwitchWithoutStalePrograms) {
  std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  uint32_t indices[]{0, 1, 2};
  Mesh<> mesh(3, 3, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get(), glm::vec3(0), glm::vec3(0.3f, 0.5f, 0.7f));
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&entity);
  scene.settings.samples_per_dispatch = 16;
  scene.settings.max_bounces = 1;
  const auto view = glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0));
  sparkium::CameraPinhole pinhole(core.get(), view, glm::radians(45.0f), 1.0f);
  sparkium::CameraThinLens lens(core.get(), view, pinhole.fovy, pinhole.aspect);
  lens.focus_distance = 2.0f;
  auto render = [&](sparkium::Camera *camera, sparkium::RenderPipeline pipeline) {
    sparkium::Film film(core.get(), 16, 16);
    core->Render(&scene, camera, &film, pipeline);
    std::vector<glm::vec4> pixels(256);
    sparkium::DownloadImage(device.Get(), film.GetRawImage(), pixels.data());
    return pixels;
  };
  std::vector<sparkium::RenderPipeline> pipelines{sparkium::RENDER_PIPELINE_RT_FALLBACK};
  if (device->SupportRayQuery())
    pipelines.push_back(sparkium::RENDER_PIPELINE_RAY_QUERY);
  if (device->SupportRayTracingPipeline())
    pipelines.push_back(sparkium::RENDER_PIPELINE_RAY_TRACING);
  for (auto pipeline : pipelines) {
    SCOPED_TRACE(int(pipeline));
    const auto reference = render(&pinhole, pipeline);
    lens.aperture_radius = 0;
    EXPECT_EQ(render(&lens, pipeline), reference);
    lens.aperture_radius = 0.5f;
    EXPECT_NE(render(&lens, pipeline), reference);
    EXPECT_EQ(render(&pinhole, pipeline), reference);
  }
}

TEST_F(SoftwareBVHTest, LazyGraphHitShadersSurvivePipelineSwitches) {
  if (!device->SupportRayTracingPipeline() || !device->SupportRayQuery())
    GTEST_SKIP() << "requires native RT and ray query";
  sparkium::CodeLines code(R"(
GraphSurface EvaluateShaderGraph(HitRecord hit, float3 direction, int bounce, uint ray_type,
                                 bool shadow, ByteAddressBuffer material) {
  GraphSurface surface = (GraphSurface)0;
  surface.normal = hit.normal;
  surface.opacity = 1.0f; surface.shadow_opacity = -1.0f;
  surface.ior = 1.45f; surface.roughness = 0.5f;
  surface.emission = float3(0.2f, 0.4f, 0.6f);
  return surface;
}
)");
  sparkium::MaterialShaderGraph material(core.get(), code, {});
  auto *graph =
      dynamic_cast<sparkium::raytracing::MaterialShaderGraph *>(sparkium::raytracing::DedicatedCast(&material));
  ASSERT_NE(graph, nullptr);
  EXPECT_EQ(graph->RenderClosestHitShader(), nullptr);
  uint32_t indices[]{0, 1, 2};
  std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}};
  Mesh<> mesh(3, 3, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&entity);
  scene.settings.samples_per_dispatch = 1;
  scene.settings.max_bounces = 1;
  sparkium::CameraPinhole camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                                 glm::radians(30.0f), 1.0f);
  std::vector<glm::vec4> reference(256), pixels(256);
  bool first_frame = true;
  for (auto pipeline : {sparkium::RENDER_PIPELINE_RAY_QUERY, sparkium::RENDER_PIPELINE_RAY_TRACING,
                        sparkium::RENDER_PIPELINE_RAY_QUERY, sparkium::RENDER_PIPELINE_RAY_TRACING}) {
    sparkium::Film film(core.get(), 16, 16);
    core->Render(&scene, &camera, &film, pipeline);
    sparkium::DownloadImage(device.Get(), film.GetRawImage(), pixels.data());
    if (first_frame) {
      first_frame = false;
      reference = pixels;
      EXPECT_EQ(graph->RenderClosestHitShader(), nullptr);
    } else {
      EXPECT_NE(graph->RenderClosestHitShader(), nullptr);
      EXPECT_NE(graph->ShadowClosestHitShader(), nullptr);
      EXPECT_NE(graph->ShadowAnyHitShader(), nullptr);
      for (size_t i = 0; i < pixels.size(); ++i)
        for (int c = 0; c < 3; ++c)
          EXPECT_NEAR(pixels[i][c], reference[i][c], 1e-6f);
    }
  }
  EXPECT_GT(reference[136].z, 0.0f);
}

TEST_F(SoftwareBVHTest, GeometryLightProgramsShareOnlyEquivalentSources) {
  auto *rt_core = sparkium::raytracing::DedicatedCast(core.get());
  auto vfs = rt_core->GetShadersVFS();
  sparkium::CodeLines geometry(vfs, "geometry/mesh/geometry_sampler.slang");
  sparkium::CodeLines diffuse(vfs, "material/lambertian/evaluator.slang");
  sparkium::CodeLines emission(vfs, "material/light/evaluator.slang");
  auto *first = rt_core->GetGeometryLightPowerProgram(geometry, diffuse);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, rt_core->GetGeometryLightPowerProgram(geometry, diffuse));
  EXPECT_NE(first, rt_core->GetGeometryLightPowerProgram(geometry, emission));
}

TEST_F(SoftwareBVHTest, HDRFilmDevelopmentPreservesHighlightsAndAccumulation) {
  sparkium::Film film(core.get(), 9, 3);
  std::vector<glm::vec4> source(27, glm::vec4(4.0f, 0.25f, -1.0f, 1.0f));
  source.back() = glm::vec4(100000.0f, 1.0f, 0.0f, 1.0f);
  sparkium::UploadImage(device.Get(), film.GetRawImage(), source.data());
  film.info.accumulated_samples = 17;
  film.info.exposure = 1.0f;
  film.info.gamma = 1.0f;
  film.info.contrast = 1.0f;
  int resets = 0;
  film.RegisterResetCallback([&]() { ++resets; });
  rhi::ObjPtr<rhi::IImage> sdr, hdr;
  (sdr = device->CreateImage({9, 3, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM}));
  (hdr = device->CreateImage({9, 3, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT}));
  EXPECT_THROW(film.Develop(sdr.Get(), true), std::invalid_argument);
  for (int transform : {0, 1, 2}) {
    film.info.view_transform = transform;
    film.Develop(sdr.Get());
    std::vector<uint8_t> before(27 * 4), after(27 * 4);
    sparkium::DownloadImage(device.Get(), sdr.Get(), before.data());
    film.Develop(hdr.Get(), true);
    std::vector<glm::vec4> actual(27);
    sparkium::DownloadImage(device.Get(), hdr.Get(), actual.data());
    for (size_t i = 0; i + 1 < actual.size(); ++i) {
      if (transform == 2) {
        EXPECT_GT(actual[i].x, 1.0f);
        EXPECT_GT(actual[i].y, 0.0f);
      } else {
        EXPECT_NEAR(actual[i].x, 8.0f, 1e-4f);
        EXPECT_NEAR(actual[i].y, 0.5f, 1e-5f);
      }
      EXPECT_EQ(actual[i].z, 0.0f);
      EXPECT_EQ(actual[i].w, 1.0f);
    }
    EXPECT_NEAR(actual.back().x, 65504.0f, 0.01f);
    film.Develop(sdr.Get());
    sparkium::DownloadImage(device.Get(), sdr.Get(), after.data());
    EXPECT_EQ(before, after);
    EXPECT_EQ(film.info.accumulated_samples, 17);
    EXPECT_EQ(film.info.view_transform, transform);
  }
  std::vector<glm::vec4> unchanged(27);
  sparkium::DownloadImage(device.Get(), film.GetRawImage(), unchanged.data());
  EXPECT_EQ(unchanged, source);
  EXPECT_EQ(resets, 0);
}

TEST_F(SoftwareBVHTest, HDRArtisticGradeMatchesSDRMidtonesAndPreservesHighlights) {
  sparkium::Film film(core.get(), 8, 1);
  const std::vector<glm::vec4> source{{0.02f, 0.08f, 0.2f, 1.0f},     {0.1f, 0.4f, 0.8f, 1.0f},
                                      {0.999f, 0.999f, 0.999f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f},
                                      {1.001f, 1.001f, 1.001f, 1.0f}, {4.0f, 2.0f, 1.0f, 1.0f},
                                      {16.0f, 8.0f, 4.0f, 1.0f},      {100000.0f, 0.0f, -1.0f, 1.0f}};
  sparkium::UploadImage(device.Get(), film.GetRawImage(), source.data());
  rhi::ObjPtr<rhi::IImage> sdr, hdr;
  ASSERT_TRUE((sdr = device->CreateImage({8, 1, rhi::IMAGE_FORMAT_R8G8B8A8_UNORM})));
  ASSERT_TRUE((hdr = device->CreateImage({8, 1, rhi::IMAGE_FORMAT_R32G32B32A32_SFLOAT})));
  auto encode = [](float x) { return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f; };
  for (int transform : {0, 1, 2}) {
    film.info.view_transform = transform;
    film.info.gamma = 1.0f;
    film.info.contrast = 1.0f;
    film.Develop(hdr.Get(), true);
    std::vector<glm::vec4> neutral(8), graded(8);
    sparkium::DownloadImage(device.Get(), hdr.Get(), neutral.data());
    for (auto grade : {glm::vec2(1.15f, 1.0f), glm::vec2(1.0f, 1.2f), glm::vec2(1.15f, 1.2f), glm::vec2(0.1f, 4.0f)}) {
      film.info.gamma = grade.x;
      film.info.contrast = grade.y;
      film.Develop(hdr.Get(), true);
      sparkium::DownloadImage(device.Get(), hdr.Get(), graded.data());
      EXPECT_NE(graded[1].y, neutral[1].y);
      EXPECT_GT(graded[5].x, 1.0f);
      EXPECT_GE(graded[6].x, graded[5].x);
      if (grade.x >= 1.0f)
        EXPECT_GT(graded[6].x, graded[5].x);
      for (const auto &pixel : graded) {
        for (int channel = 0; channel < 3; ++channel) {
          EXPECT_TRUE(std::isfinite(pixel[channel]));
          EXPECT_GE(pixel[channel], 0.0f);
          EXPECT_LE(pixel[channel], 65504.0f);
        }
        EXPECT_EQ(pixel.w, 1.0f);
      }
      if (transform == 2 && grade.x > 1.0f) {
        film.Develop(sdr.Get());
        std::vector<uint8_t> bytes(32);
        sparkium::DownloadImage(device.Get(), sdr.Get(), bytes.data());
        for (int pixel = 0; pixel < 4; ++pixel)
          for (int channel = 0; channel < 3; ++channel)
            EXPECT_NEAR(encode(graded[pixel][channel]), bytes[pixel * 4 + channel] / 255.0f, 1.0f / 255.0f);
        EXPECT_LT(graded[2].x, graded[3].x);
        EXPECT_LT(graded[3].x, graded[4].x);
        EXPECT_NEAR(graded[3].x - graded[2].x, graded[4].x - graded[3].x, 1e-5f);
      }
    }
  }
}

TEST(GraphicsCoreCreation, UnsupportedAPIsDoNotFallBack) {
  for (auto api : {graphics::BACKEND_API_METAL, graphics::BACKEND_API_D3D12, graphics::BACKEND_API_VULKAN,
                   static_cast<graphics::BackendAPI>(99)}) {
    if (graphics::SupportBackendAPI(api))
      continue;
    std::unique_ptr<graphics::Core> core;
    EXPECT_EQ(graphics::CreateCore(api, graphics::Core::Settings{}, &core), -1);
    EXPECT_EQ(core, nullptr);
  }
}

TEST_F(SoftwareBVHTest, LightSamplingIsIndependentOfEntityAddresses) {
  std::vector<std::unique_ptr<sparkium::EntityPointLight>> owned;
  std::vector<std::pair<sparkium::raytracing::Entity *, sparkium::EntityPointLight *>> lights;
  for (int i = 0; i < 4; ++i) {
    owned.push_back(std::make_unique<sparkium::EntityPointLight>(core.get()));
    lights.emplace_back(sparkium::raytracing::DedicatedCast(owned.back().get()), owned.back().get());
  }

  std::sort(lights.begin(), lights.end(),
            [](const auto &a, const auto &b) { return std::less<sparkium::raytracing::Entity *>{}(a.first, b.first); });
  // Equivalent light lists, deliberately opposite address ordering in the
  // renderer's component cache. Sampling must follow insertion, not addresses.
  for (int i = 0; i < 4; ++i) {
    const bool red = i == 0 || i == 2;
    lights[i].second->position = red ? glm::vec3(-1, 1, 2) : glm::vec3(2, 0, 1);
    lights[i].second->color = red ? glm::vec3(1, 0.1f, 0.2f) : glm::vec3(0.1f, 0.2f, 1);
    lights[i].second->strength = red ? 10.0f : 40.0f;
  }

  std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}};
  uint32_t indices[]{0, 1, 2};
  Mesh<> mesh(3, 3, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get(), glm::vec3(0.6f));
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene first(core.get()), second(core.get());
  for (auto *scene : {&first, &second}) {
    scene->AddEntity(&entity);
    scene->settings.samples_per_dispatch = 1;
    scene->settings.max_bounces = 2;
  }
  first.AddEntity(lights[0].second);
  first.AddEntity(lights[3].second);
  second.AddEntity(lights[2].second);
  second.AddEntity(lights[1].second);
  first.AddEntity(lights[0].second);  // Duplicate adds must not register twice.
  first.SetEntityActive(lights[0].second, false);
  first.SetEntityActive(lights[0].second, true);
  sparkium::CameraPinhole camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                                 glm::radians(45.0f), 1.0f);
  for (auto pipeline : {sparkium::RENDER_PIPELINE_RT_FALLBACK, sparkium::RENDER_PIPELINE_RAY_QUERY,
                        sparkium::RENDER_PIPELINE_RAY_TRACING, sparkium::RENDER_PIPELINE_RASTERIZATION}) {
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_QUERY && !device->SupportRayQuery())
      continue;
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_TRACING && !device->SupportRayTracingPipeline())
      continue;
    SCOPED_TRACE(static_cast<int>(pipeline));
    sparkium::Film a(core.get(), 32, 32), b(core.get(), 32, 32);
    core->Render(&first, &camera, &a, pipeline);
    core->Render(&second, &camera, &b, pipeline);
    std::vector<glm::vec4> left(1024), right(1024);
    sparkium::DownloadImage(device.Get(), a.GetRawImage(), left.data());
    sparkium::DownloadImage(device.Get(), b.GetRawImage(), right.data());
    double radiance = 0;
    for (size_t i = 0; i < left.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(left[i][c], right[i][c], 1e-6f);
        radiance += left[i][c];
      }
    EXPECT_GT(radiance, 1.0);
  }
}

TEST_F(SoftwareBVHTest, NonblockingEmittersDoNotHideShadowOccluders) {
  uint32_t indices[]{0, 1, 2, 0, 2, 3};
  std::vector<Vector3<float>> receiver_positions{{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}};
  std::vector<Vector3<float>> wall_positions{{0, -10, -10}, {0, 10, -10}, {0, 10, 10}, {0, -10, 10}};
  Mesh<> receiver_mesh(4, 6, indices, receiver_positions.data());
  Mesh<> wall_mesh(4, 6, indices, wall_positions.data());
  sparkium::GeometryMesh receiver_geometry(core.get(), receiver_mesh), wall_geometry(core.get(), wall_mesh);
  sparkium::MaterialLambertian diffuse(core.get(), glm::vec3(0.7f));
  sparkium::MaterialLight invisible(core.get(), glm::vec3(0), false, false, false);
  sparkium::EntityGeometryMaterial receiver(core.get(), &receiver_geometry, &diffuse);
  sparkium::EntityGeometryMaterial emitter(core.get(), &wall_geometry, &invisible,
                                           glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(0.75f, 0, 0))));
  sparkium::EntityGeometryMaterial occluder(core.get(), &wall_geometry, &diffuse,
                                            glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(1.5f, 0, 0))));
  sparkium::EntityPointLight light(core.get(), glm::vec3(3, 0, 3), glm::vec3(1), 100.0f);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&receiver);
  scene.AddEntity(&emitter);
  scene.AddEntity(&occluder);
  scene.AddEntity(&light);
  scene.settings.samples_per_dispatch = 16;
  scene.settings.max_bounces = 1;
  scene.settings.alpha_shadow = true;
  sparkium::CameraPinhole camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                                 glm::radians(20.0f), 1.0f);
  for (auto pipeline : {sparkium::RENDER_PIPELINE_RT_FALLBACK, sparkium::RENDER_PIPELINE_RAY_QUERY,
                        sparkium::RENDER_PIPELINE_RAY_TRACING}) {
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_QUERY && !device->SupportRayQuery())
      continue;
    if (pipeline == sparkium::RENDER_PIPELINE_RAY_TRACING && !device->SupportRayTracingPipeline())
      continue;
    for (int mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(testing::Message() << "pipeline=" << pipeline << ", mode=" << mode);
      scene.SetEntityActive(&occluder, mode == 0);
      invisible.block_ray = mode == 2;
      sparkium::Film film(core.get(), 32, 32);
      core->Render(&scene, &camera, &film, pipeline);
      std::vector<glm::vec4> pixels(1024);
      sparkium::DownloadImage(device.Get(), film.GetRawImage(), pixels.data());
      double radiance = 0;
      for (const auto &pixel : pixels)
        for (int c = 0; c < 3; ++c) {
          ASSERT_TRUE(std::isfinite(pixel[c]));
          radiance += pixel[c];
          if (mode != 1)
            EXPECT_NEAR(pixel[c], 0.0f, 1e-6f);
        }
      if (mode == 1)
        EXPECT_GT(radiance, 1.0);
    }
  }
}

TEST_F(SoftwareBVHTest, RayQueryCapabilityMatchesDevice) {
#if defined(LONGMARCH_VULKAN_ENABLED)
  if (auto *vk = dynamic_cast<rhi::backend::vulkan::VulkanDevice *>(device.Get())) {
    VkPhysicalDeviceRayQueryFeaturesKHR query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &query};
    vkGetPhysicalDeviceFeatures2(vk->NativePhysicalDevice(), &features);
    EXPECT_EQ(device->SupportRayQuery(), query.rayQuery != 0);
    if (device->SupportRayQuery())
      EXPECT_NE(vkGetDeviceProcAddr(vk->NativeDevice(), "vkCmdBuildAccelerationStructuresKHR"), nullptr);
  }
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
  if (auto *dx = dynamic_cast<rhi::backend::d3d12::D3D12Device *>(device.Get())) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
    auto result = dx->NativeDevice()->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options));
    EXPECT_EQ(device->SupportRayQuery(), SUCCEEDED(result) && options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);
  }
#endif
}

Hit Oracle(const Ray &ray, const std::vector<Vector3<float>> &positions, const std::vector<glm::mat4x3> &transforms) {
  Hit hit{};
  hit.distance = ray.t_max;
  hit.instance = hit.primitive = ~0u;
  for (uint32_t instance = 0; instance < transforms.size(); ++instance) {
    for (uint32_t triangle = 0; triangle < positions.size() / 3; ++triangle) {
      glm::dvec3 p[3];
      for (int v = 0; v < 3; ++v) {
        auto vertex = positions[triangle * 3 + v];
        p[v] = glm::dmat4(transforms[instance]) * glm::dvec4(vertex.x(), vertex.y(), vertex.z(), 1.0);
      }
      const glm::dvec3 e1 = p[1] - p[0], e2 = p[2] - p[0], direction(ray.direction);
      const glm::dvec3 cross = glm::cross(direction, e2);
      double determinant = glm::dot(e1, cross);
      if (std::abs(determinant) < 1e-12)
        continue;
      glm::dvec3 delta = glm::dvec3(ray.origin) - p[0];
      double u = glm::dot(delta, cross) / determinant;
      glm::dvec3 q = glm::cross(delta, e1);
      double v = glm::dot(direction, q) / determinant;
      double t = glm::dot(e2, q) / determinant;
      if (u < 0 || v < 0 || u + v > 1 || t < ray.t_min || t >= hit.distance)
        continue;
      hit = {static_cast<float>(t), instance, triangle, 1, static_cast<float>(u), static_cast<float>(v), {0, 0}};
    }
  }
  return hit;
}

class SoftwareBVHSizeTest : public SoftwareBVHTest, public testing::WithParamInterface<std::tuple<int, bool>> {};

TEST_P(SoftwareBVHSizeTest, ComputeConstructionAndTraversalMatchDoublePrecisionOracle) {
  auto [triangle_count, ray_query] = GetParam();
  if (ray_query && !device->SupportRayQuery())
    GTEST_SKIP() << "native ray query unavailable";
  // Five triangles: a non-power-of-two tree, duplicate centroids and a degenerate leaf.
  std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0},  {0, 1, 0},  {-1, -1, -1}, {1, -1, -1},
                                        {0, 1, -1},  {-2, -1, 0}, {-2, 1, 0}, {-2, 0, 2},   {-1, -1, 0},
                                        {0, 1, 0},   {1, -1, 0},  {0, 0, 0},  {0, 0, 0},    {0, 0, 0}};
  std::mt19937 geometry_random(951);
  std::uniform_real_distribution<float> coordinate(-3.0f, 3.0f);
  while (positions.size() < triangle_count * 3) {
    Vector3<float> center(coordinate(geometry_random), coordinate(geometry_random), coordinate(geometry_random));
    positions.push_back(center + Vector3<float>(-0.3f, -0.2f, 0));
    positions.push_back(center + Vector3<float>(0.3f, -0.2f, 0.1f));
    positions.push_back(center + Vector3<float>(0, 0.3f, -0.1f));
  }
  positions.resize(triangle_count * 3);
  std::vector<uint32_t> indices(positions.size());
  std::iota(indices.begin(), indices.end(), 0);
  Mesh<> mesh(positions.size(), indices.size(), indices.data(), positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get());
  sparkium::raytracing::GeometryMesh rt_geometry(geometry);
  sparkium::raytracing::MaterialLambertian rt_material(material);
  sparkium::raytracing::SoftwarePipeline pipeline(sparkium::raytracing::DedicatedCast(core.get()), ray_query);
  std::vector<rhi::IBuffer *> buffers{rt_geometry.Buffer()};

  std::mt19937 random(7411);
  std::uniform_real_distribution<float> position(-5.0f, 5.0f);
  // Match renderer self-intersection exclusion; exact t=0 acceptance differs across traversal APIs.
  std::vector<Ray> rays{{{0, 0, 4}, 0, {0, 0, -1}, 100},      {{0, 0, 4}, 0, {0, 0, 1}, 100},
                        {{0, 0, 0}, 0.001f, {0, 0, -1}, 100}, {{0, 0, 4}, 0, {0, 0, -1}, 1},
                        {{10, 0, 0}, 0, {0, 1, 0}, 100},      {{0, 0, 4}, 4.1f, {0, 0, -1}, 100}};
  for (int i = 0; i < 1024; ++i) {
    glm::vec3 origin(position(random), position(random), 4.0f);
    glm::vec3 target(position(random), position(random), -1.0f);
    rays.push_back({origin, 0.001f, glm::normalize(target - origin), 100.0f});
  }

  rhi::ObjPtr<rhi::IBuffer> ray_buffer, output;
  (ray_buffer = device->CreateBuffer({16 + rays.size() * sizeof(Ray)}));
  uint32_t count = rays.size();
  sparkium::UploadBuffer(device.Get(), ray_buffer.Get(), &count, sizeof(count));
  sparkium::UploadBuffer(device.Get(), ray_buffer.Get(), rays.data(), rays.size() * sizeof(Ray), 16);
  (output = device->CreateBuffer({rays.size() * sizeof(Hit)}));

  auto vfs = core->GetShadersVFS();
  vfs.WriteFile("bvh_test.slang", R"(
#define SOFTWARE_EXTERNAL_BINDINGS
#ifdef NATIVE_QUERY
RHI_ACCELERATION_STRUCTURE(0, 0, query_scene);
#else
RHI_BYTE_ADDRESS_BUFFER(0, 0, software_nodes);
#endif
RHI_BYTE_ADDRESS_BUFFER(0, 1, software_instances);
RHI_BYTE_ADDRESS_BUFFER_ARRAY(0, 2, data_buffers);
RHI_BYTE_ADDRESS_BUFFER(0, 3, rays);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 4, results);
#ifdef NATIVE_QUERY
#include "ray_query/traversal.slang"
#else
#include "software/traversal.slang"
#endif
[numthreads(64, 1, 1)] void Main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rays.Load(0)) return;
  uint offset = 16 + id.x * 32;
  RayDesc ray;
  ray.Origin = asfloat(rays.Load3(offset)); ray.TMin = asfloat(rays.Load(offset + 12));
  ray.Direction = asfloat(rays.Load3(offset + 16)); ray.TMax = asfloat(rays.Load(offset + 28));
  SoftwareHit hit;
  bool found = InlineIntersect(ray, false, hit);
  results.Store4(id.x * 32, uint4(asuint(hit.distance), hit.instance, hit.primitive, uint(found)));
  results.Store4(id.x * 32 + 16, uint4(asuint(hit.barycentric), 0, 0));
}
)");
  rhi::ObjPtr<rhi::IShader> shader;
  std::vector<std::string> args{"-I."};
  if (ray_query)
    args.push_back("-DNATIVE_QUERY");
  shader = device->CreateShader("bvh_test.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &vfs, {"."},
                                ray_query ? std::vector<std::pair<std::string, std::string>>{{"NATIVE_QUERY", "1"}}
                                          : std::vector<std::pair<std::string, std::string>>{});
  ASSERT_TRUE(shader);
  rhi::ObjPtr<rhi::IComputeProgram> program;
  rhi::ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups.resize(1);
  for (uint32_t i = 0; i < 4; ++i)
    desc.pipeline_layout.groups[0].bindings.push_back(
        {i, ray_query && i == 0 ? rhi::BINDING_ACCELERATION_STRUCTURE : rhi::BINDING_BYTE_ADDRESS_BUFFER});
  desc.pipeline_layout.groups[0].bindings.push_back({4, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER});
  program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  rhi::ObjPtr<rhi::IBindingGroup> bindings = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(bindings);

  glm::mat4 mirrored =
      glm::translate(glm::mat4(1), glm::vec3(3, 0, -1)) * glm::scale(glm::mat4(1), glm::vec3(-0.7f, 1.8f, 0.4f));
  std::vector<glm::mat4x3> transforms{glm::mat4x3(1), glm::mat4x3(mirrored)};
  for (int phase = 0; phase < 4; ++phase) {
    // Refit instances, shrink the instance tree, then exercise a completely empty TLAS.
    if (phase == 1)
      transforms[0] = glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(-1, 0.5f, -2)));
    if (phase == 2)
      transforms.resize(1);
    if (phase == 3)
      transforms.clear();
    pipeline.ClearInstances();
    for (auto transform : transforms)
      pipeline.AddInstance(&rt_geometry, &rt_material, transform, 0);
    rhi::ObjPtr<rhi::ICommandContext> commands;
    commands = device->CreateCommandContext();
    sparkium::CameraPinhole camera(core.get(), glm::mat4(1), glm::radians(45.0f), 1.0f);
    pipeline.Update(commands.Get(), buffers, 1, 1, sparkium::raytracing::DedicatedCast(&camera));
    commands->CmdBindComputeProgram(program.Get());
    if (ray_query)
      bindings->BindResource(0, pipeline.AccelerationStructure());
    else
      bindings->BindResource(0, pipeline.Nodes());
    bindings->BindResource(1, pipeline.Instances());
    bindings->BindResource(2, buffers);
    bindings->BindResource(3, ray_buffer.Get());
    bindings->BindResource(4, output.Get());
    commands->CmdBindGroup(0, bindings.Get());
    commands->CmdDispatch((rays.size() + 63) / 64, 1, 1);
    EXPECT_TRUE(sparkium::Execute(device.Get(), commands.Get()));
    std::vector<Hit> actual(rays.size());
    sparkium::DownloadBuffer(device.Get(), output.Get(), actual.data(), actual.size() * sizeof(Hit));
    for (size_t i = 0; i < rays.size(); ++i) {
      SCOPED_TRACE(testing::Message() << "phase " << phase << ", ray " << i);
      const Hit expected = Oracle(rays[i], positions, transforms);
      ASSERT_EQ(actual[i].found, expected.found);
      if (expected.found) {
        EXPECT_NEAR(actual[i].distance, expected.distance, 2.0e-4f * std::max(1.0f, expected.distance));
        EXPECT_EQ(actual[i].instance, expected.instance);
        if (actual[i].primitive == expected.primitive) {
          EXPECT_NEAR(actual[i].u, expected.u, 1.0e-4f);
          EXPECT_NEAR(actual[i].v, expected.v, 1.0e-4f);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TreeSizes, SoftwareBVHSizeTest, testing::Combine(testing::Values(1, 5, 257), testing::Bool()));

class ComputeTraversalTest : public SoftwareBVHTest, public testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(TraversalModes, ComputeTraversalTest, testing::Bool());

TEST_P(ComputeTraversalTest, EmptySceneBackgroundAccumulationAndReset) {
  bool ray_query = GetParam();
  if (ray_query && !device->SupportRayQuery())
    GTEST_SKIP() << "native ray query unavailable";
  sparkium::Scene scene(core.get());
  scene.settings.samples_per_dispatch = 3;
  scene.settings.background_color = glm::vec3(0.2f, 0.4f, 0.7f);
  sparkium::CameraPinhole camera(core.get(), glm::mat4(1), glm::radians(45.0f), 17.0f / 13.0f);
  sparkium::Film film(core.get(), 17, 13);
  auto check = [&](glm::vec3 expected) {
    std::vector<glm::vec4> pixels(17 * 13);
    sparkium::DownloadImage(device.Get(), film.GetRawImage(), pixels.data());
    for (auto pixel : pixels)
      for (int c = 0; c < 3; ++c) {
        ASSERT_TRUE(std::isfinite(pixel[c]));
        EXPECT_NEAR(pixel[c], expected[c], 1e-5f);
      }
  };

  for (int frame = 1; frame <= 2; ++frame) {
    core->Render(&scene, &camera, &film,
                 ray_query ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK);
    EXPECT_EQ(film.info.accumulated_samples, frame * 3);
    check(scene.settings.background_color);
  }
  scene.settings.background_color = glm::vec3(0.1f, 0.3f, 0.9f);
  film.Reset();
  EXPECT_EQ(film.info.accumulated_samples, 0);
  core->Render(&scene, &camera, &film,
               ray_query ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK);
  EXPECT_EQ(film.info.accumulated_samples, 3);
  check(scene.settings.background_color);
  if (ray_query && device->SupportRayQuery()) {
    EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO), sparkium::RENDER_PIPELINE_RAY_QUERY);
    sparkium::FrameProfile profile(device.Get());
    profile.Begin();
    core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_AUTO);
    profile.Finish();
    EXPECT_EQ(profile.counters["native_ray_query"], 1u);
    // Auto must reuse the selected native pipeline and preserve its accumulation.
    EXPECT_EQ(film.info.accumulated_samples, 6);
    check(scene.settings.background_color);
    if (!device->SupportRayTracingPipeline()) {
      // Legacy JSON requests must select the same native query pipeline as Auto.
      EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_RAY_TRACING),
                sparkium::RENDER_PIPELINE_RAY_QUERY);
      profile.Begin();
      core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RAY_TRACING);
      profile.Finish();
      EXPECT_EQ(profile.counters["native_ray_query"], 1u);
      EXPECT_EQ(film.info.accumulated_samples, 9);
      check(scene.settings.background_color);
    } else {
      EXPECT_EQ(core->ResolveRenderPipeline(sparkium::RENDER_PIPELINE_RAY_TRACING),
                sparkium::RENDER_PIPELINE_RAY_TRACING);
    }
  }
  if (device->SupportRayQuery()) {
    // Switching traversal implementations must discard accumulation in both directions.
    for (bool query : {!ray_query, ray_query}) {
      core->Render(&scene, &camera, &film,
                   query ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK);
      EXPECT_EQ(film.info.accumulated_samples, 3);
      check(scene.settings.background_color);
    }
  }
}

TEST_P(ComputeTraversalTest, TransparentShadowLayers) {
  bool ray_query = GetParam();
  if (ray_query && !device->SupportRayQuery())
    GTEST_SKIP() << "native ray query unavailable";
  std::vector<Vector3<float>> positions{{-2, -2, 0}, {2, -2, 0}, {0, 2, 0}, {-2, -2, -1}, {2, -2, -1}, {0, 2, -1}};
  uint32_t indices[]{0, 1, 2, 3, 4, 5};
  Mesh<> mesh(6, 6, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get());
  sparkium::raytracing::GeometryMesh rt_geometry(geometry);
  sparkium::raytracing::MaterialLambertian rt_material(material);
  sparkium::raytracing::SoftwarePipeline pipeline(sparkium::raytracing::DedicatedCast(core.get()), ray_query);
  pipeline.AddInstance(&rt_geometry, &rt_material, glm::mat4x3(1), 0);
  auto vfs = core->GetShadersVFS();
  vfs.WriteFile("shadow_test.slang", R"(
#define SOFTWARE_EXTERNAL_BINDINGS
#include "common.slang"
#ifdef NATIVE_QUERY
RHI_ACCELERATION_STRUCTURE(0, 0, query_scene);
#else
RHI_BYTE_ADDRESS_BUFFER(0, 0, software_nodes);
#endif
RHI_BYTE_ADDRESS_BUFFER(0, 1, software_instances);
RHI_BYTE_ADDRESS_BUFFER_ARRAY(0, 2, data_buffers);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 3, results);
#ifdef NATIVE_QUERY
#include "ray_query/traversal.slang"
#else
#include "software/traversal.slang"
#endif
HitRecord SoftwareHitRecord(SoftwareHit hit, float3 direction) { return (HitRecord)0; }
float SoftwareShadowTransmission(uint material, HitRecord hit, float3 direction) { return 0.5f; }
#include "software/shadow.slang"
[numthreads(1, 1, 1)] void Main() {
  float3 o = float3(0, 0, 4), d = float3(0, 0, -1);
  results.Store4(0, asuint(float4(ShadowRay(o, d, 10), ShadowRayNoAlpha(o, d, 10),
                                  ShadowRay(o, d, 4.5f), ShadowRay(o, d, 3.0f))));
}
)");
  rhi::ObjPtr<rhi::IShader> shader;
  std::vector<std::string> args{"-I."};
  if (ray_query)
    args.push_back("-DNATIVE_QUERY");
  shader = device->CreateShader("shadow_test.slang", "Main", rhi::SHADER_STAGE_COMPUTE, &vfs, {"."},
                                ray_query ? std::vector<std::pair<std::string, std::string>>{{"NATIVE_QUERY", "1"}}
                                          : std::vector<std::pair<std::string, std::string>>{});
  ASSERT_TRUE(shader);
  rhi::ObjPtr<rhi::IComputeProgram> program;
  rhi::ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups.resize(1);
  for (uint32_t i = 0; i < 3; ++i)
    desc.pipeline_layout.groups[0].bindings.push_back(
        {i, ray_query && i == 0 ? rhi::BINDING_ACCELERATION_STRUCTURE : rhi::BINDING_BYTE_ADDRESS_BUFFER});
  desc.pipeline_layout.groups[0].bindings.push_back({3, rhi::BINDING_RW_BYTE_ADDRESS_BUFFER});
  program = device->CreateComputeProgram(desc);
  ASSERT_TRUE(program);
  rhi::ObjPtr<rhi::IBindingGroup> bindings = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  ASSERT_TRUE(bindings);
  rhi::ObjPtr<rhi::IBuffer> output;
  (output = device->CreateBuffer({16}));
  rhi::ObjPtr<rhi::ICommandContext> commands;
  commands = device->CreateCommandContext();
  sparkium::CameraPinhole camera(core.get(), glm::mat4(1), glm::radians(45.0f), 1.0f);
  pipeline.Update(commands.Get(), {rt_geometry.Buffer()}, 1, 1, sparkium::raytracing::DedicatedCast(&camera));
  commands->CmdBindComputeProgram(program.Get());
  if (ray_query)
    bindings->BindResource(0, pipeline.AccelerationStructure());
  else
    bindings->BindResource(0, pipeline.Nodes());
  bindings->BindResource(1, pipeline.Instances());
  bindings->BindResource(2, rt_geometry.Buffer());
  bindings->BindResource(3, output.Get());
  commands->CmdBindGroup(0, bindings.Get());
  commands->CmdDispatch(1, 1, 1);
  EXPECT_TRUE(sparkium::Execute(device.Get(), commands.Get()));
  float actual[4];
  sparkium::DownloadBuffer(device.Get(), output.Get(), actual, sizeof(actual));
  EXPECT_FLOAT_EQ(actual[0], 0.25f);
  EXPECT_FLOAT_EQ(actual[1], 0.0f);
  EXPECT_FLOAT_EQ(actual[2], 0.5f);
  EXPECT_FLOAT_EQ(actual[3], 1.0f);
}

TEST_F(SoftwareBVHTest, SharedShadersCompileForNativeRayTracingAndCompute) {
  auto vfs = core->GetShadersVFS();
  for (bool spirv : {false, true}) {
    // DXIL's downstream compiler is a D3D12 dependency, not a Metal requirement.
    if (!spirv && !graphics::SupportBackendAPI(graphics::BACKEND_API_D3D12))
      continue;
    std::vector<std::string> args{"-I.", "-warnings-as-errors", "all"};
    if (spirv)
      args.insert(args.end(), {"-target", "spirv", "-profile", "spirv_1_5", "-fvk-use-dx-layout"});
    auto compile = [&](const char *file, const char *entry, const char *target) {
      SCOPED_TRACE(testing::Message() << file << " / " << entry << ", SPIR-V=" << spirv);
      using namespace rhi;
      PipelineLayoutDesc layout;
      auto &bindings = layout.groups.emplace_back().bindings;
      const std::string filename = file;
      if (filename == "software/build.slang") {
        bindings = {{0, BINDING_RW_BYTE_ADDRESS_BUFFER},
                    {1, BINDING_BYTE_ADDRESS_BUFFER},
                    {2, BINDING_UNIFORM_BUFFER},
                    {3, BINDING_RW_BYTE_ADDRESS_BUFFER},
                    {4, BINDING_BYTE_ADDRESS_BUFFER}};
      } else if (filename == "blelloch_scan.slang") {
        bindings = {{0, BINDING_RW_BYTE_ADDRESS_BUFFER}, {1, BINDING_UNIFORM_BUFFER}};
      } else if (filename == "film2img.slang") {
        bindings = {{0, BINDING_TEXTURE2D}, {1, BINDING_TEXTURE2D}, {2, BINDING_RW_TEXTURE2D}};
      } else if (filename == "material/principled/pixel_shader.slang") {
        bindings = {{2, BINDING_BYTE_ADDRESS_BUFFER}, {3, BINDING_TEXTURE2D, 6}, {4, BINDING_SAMPLER}};
      } else {
        bindings = {
            {0, BINDING_RW_TEXTURE2D},        {1, BINDING_RW_TEXTURE2D},        {2, BINDING_ACCELERATION_STRUCTURE},
            {3, BINDING_UNIFORM_BUFFER},      {4, BINDING_BYTE_ADDRESS_BUFFER}, {5, BINDING_BYTE_ADDRESS_BUFFER},
            {6, BINDING_BYTE_ADDRESS_BUFFER}, {7, BINDING_BYTE_ADDRESS_BUFFER}, {8, BINDING_BYTE_ADDRESS_BUFFER},
            {9, BINDING_BYTE_ADDRESS_BUFFER}, {10, BINDING_TEXTURE2D},          {11, BINDING_TEXTURE2D},
            {12, BINDING_SAMPLER, 2}};
      }
      ShaderStage stage = SHADER_STAGE_COMPUTE;
      if (std::string(target).rfind("ps", 0) == 0)
        stage = SHADER_STAGE_FRAGMENT;
      if (std::string(target).rfind("lib", 0) == 0) {
        stage = SHADER_STAGE_CALLABLE;
        const std::string name = entry;
        if (name.find("ClosestHit") != std::string::npos)
          stage = SHADER_STAGE_CLOSEST_HIT;
        else if (name.find("AnyHit") != std::string::npos)
          stage = SHADER_STAGE_ANY_HIT;
        else if (name.find("Miss") != std::string::npos)
          stage = SHADER_STAGE_MISS;
        else if (name == "Main")
          stage = SHADER_STAGE_RAY_GENERATION;
      }
      ObjPtr<IShader> shader = device->CreateShader(file, entry, stage, &vfs, {"."});
      ASSERT_TRUE(shader);
      std::vector<uint8_t> code;
      auto target_api = spirv ? BACKEND_API_VULKAN : BACKEND_API_D3D12;
      EXPECT_TRUE(rhi::backend::CompileSlang(shader.Get(), rhi::backend::ShaderMacroHeader(layout, target_api),
                                             target_api, code, &layout));
      EXPECT_FALSE(code.empty());
    };
    for (auto entry : {"Main", "MissMain", "ShadowMiss"})
      compile("raygen.slang", entry, "lib_6_5");
    compile("camera/pinhole.slang", "CameraPinhole", "lib_6_5");
    compile("camera/thin_lens.slang", "CameraThinLens", "lib_6_5");
    for (auto material : {"lambertian", "light", "principled", "specular"}) {
      vfs.WriteFile("material_sampler.slang",
                    sparkium::CodeLines(vfs, std::string("material/") + material + "/sampler.slang"));
      compile("geometry/mesh/hit_group.slang", "RenderClosestHit", "lib_6_5");
      compile("geometry/mesh/hit_group.slang", "ShadowClosestHit", "lib_6_5");
    }
    sparkium::CodeLines graph(vfs, "material/shader_graph/sampler.slang");
    graph.InsertAfter(sparkium::CodeLines(R"(
GraphSurface EvaluateShaderGraph(HitRecord hit, float3 direction, int bounce, uint ray_type,
                                 bool shadow, ByteAddressBuffer material) {
  GraphSurface surface = (GraphSurface)0;
  surface.normal = hit.normal;
  surface.opacity = 0.5f; surface.shadow_opacity = -1.0f;
  surface.ior = 1.45f; surface.roughness = 0.5f;
  return surface;
}
)"),
                      "// SHADER_GRAPH_IMPLEMENTATION");
    vfs.WriteFile("material_sampler.slang", graph);
    for (auto entry : {"RenderClosestHit", "ShadowClosestHit", "ShadowAnyHit"})
      compile("geometry/mesh/hit_group.slang", entry, "lib_6_5");
    for (auto entry : {"InitLeaves", "ReduceNodes", "MortonKeys", "BitonicSort", "SortLeaves"})
      compile("software/build.slang", entry, "cs_6_0");
    for (auto entry : {"BlellochUpSweep", "BlellochDownSweep"})
      compile("blelloch_scan.slang", entry, "cs_6_0");
    compile("film2img.slang", "Main", "cs_6_0");
    compile("material/principled/pixel_shader.slang", "PSMain", "ps_6_0");
  }
}

TEST_F(SoftwareBVHTest, RasterPointLightsLeaveEmptyBackgroundUnchanged) {
  sparkium::Scene scene(core.get());
  scene.settings.ambient_light = glm::vec3(0.2f);
  sparkium::EntityPointLight light(core.get(), glm::vec3(0, 0, 1), glm::vec3(1), 100.0f);
  scene.AddEntity(&light);
  sparkium::CameraPinhole camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                                 glm::radians(45.0f), 17.0f / 13.0f);
  sparkium::Film film(core.get(), 17, 13);
  core->Render(&scene, &camera, &film, sparkium::RENDER_PIPELINE_RASTERIZATION);
  std::vector<glm::vec4> pixels(17 * 13);
  sparkium::DownloadImage(device.Get(), film.GetRawImage(), pixels.data());
  for (const auto &pixel : pixels)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(pixel[c], 0.2f, 1e-6f);
}

TEST_F(SoftwareBVHTest, HardwareImageParity) {
  if (!device->SupportRayTracingPipeline())
    GTEST_SKIP() << "requires a hardware ray tracing device";
  std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
  uint32_t indices[]{0, 1, 2};
  Mesh<> mesh(3, 3, indices, positions.data());
  sparkium::GeometryMesh geometry(core.get(), mesh);
  sparkium::MaterialLambertian material(core.get(), glm::vec3(0.4f), glm::vec3(0.3f, 0.5f, 0.7f));
  sparkium::EntityGeometryMaterial entity(core.get(), &geometry, &material);
  sparkium::Scene scene(core.get());
  scene.AddEntity(&entity);
  scene.settings.samples_per_dispatch = 16;
  scene.settings.max_bounces = 4;
  scene.settings.background_color = glm::vec3(0.1f);
  sparkium::CameraPinhole camera(core.get(), glm::lookAt(glm::vec3(0, 0, 4), glm::vec3(0), glm::vec3(0, 1, 0)),
                                 glm::radians(45.0f), 1.0f);
  sparkium::Film software(core.get(), 32, 32), hardware(core.get(), 32, 32);
  core->Render(&scene, &camera, &software, sparkium::RENDER_PIPELINE_RT_FALLBACK);
  core->Render(&scene, &camera, &hardware, sparkium::RENDER_PIPELINE_RAY_TRACING);
  std::vector<glm::vec4> a(1024), b(1024);
  sparkium::DownloadImage(device.Get(), software.GetRawImage(), a.data());
  sparkium::DownloadImage(device.Get(), hardware.GetRawImage(), b.data());
  double error = 0;
  for (size_t i = 0; i < a.size(); ++i)
    for (int c = 0; c < 3; ++c) {
      ASSERT_TRUE(std::isfinite(a[i][c]) && std::isfinite(b[i][c]));
      error += std::abs(a[i][c] - b[i][c]);
    }
  EXPECT_LT(error / (a.size() * 3), 0.01);
}
}  // namespace
