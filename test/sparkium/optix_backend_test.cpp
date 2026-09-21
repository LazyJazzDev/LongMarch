#include <gtest/gtest.h>
#include <long_march.h>

#include <array>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>

using namespace grassland;

TEST(CudaRayTracing, HardwareHitsTransformsMasksAndEmptyScene) {
#ifndef SPARKIUM_OPTIX_ENABLED
  GTEST_SKIP() << "OptiX not built";
#endif
  std::unique_ptr<sparkium::backend::Device> core;
  ASSERT_EQ(sparkium::CreateDevice(sparkium::RenderBackend::CUDA, {2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(true), 0);
  ASSERT_TRUE(core->DeviceRayTracingSupport());
  const float positions[]{-1, -1, 0, 1, -1, 0, 0, 1, 0};
  const uint32_t indices[]{0, 1, 2};
  std::unique_ptr<graphics::Buffer> vertices, triangles, results, parameters;
  core->CreateBuffer(sizeof(positions), graphics::BUFFER_TYPE_STATIC, &vertices);
  core->CreateBuffer(sizeof(indices), graphics::BUFFER_TYPE_STATIC, &triangles);
  core->CreateBuffer(4 * 20, graphics::BUFFER_TYPE_STATIC, &results);
  core->CreateBuffer(16, graphics::BUFFER_TYPE_STATIC, &parameters);
  vertices->UploadData(positions, sizeof(positions));
  triangles->UploadData(indices, sizeof(indices));
  std::unique_ptr<graphics::AccelerationStructure> blas, tlas;
  ASSERT_EQ(core->CreateBottomLevelAccelerationStructure(vertices->Range(), triangles->Range(), 3, 12, 1,
                                                         graphics::RAYTRACING_GEOMETRY_FLAG_NONE, &blas),
            0);
  EXPECT_THROW(core->CreateBottomLevelAccelerationStructure(vertices->Range(), triangles->Range(), 3, 12, 2,
                                                            graphics::RAYTRACING_GEOMETRY_FLAG_NONE, &tlas),
               std::out_of_range);
  auto transform = [](float x, float z) {
    return glm::mat4x3(glm::translate(glm::mat4(1), glm::vec3(x, 0, z)) *
                       glm::scale(glm::mat4(1), glm::vec3(2, 0.5f, -1)));
  };
  std::vector<graphics::RayTracingInstance> instances{blas->MakeInstance(glm::mat4x3(1), 17, 1),
                                                      blas->MakeInstance(transform(3, 1), 31, 1),
                                                      blas->MakeInstance(transform(0, 0.5f), 18, 2)};
  ASSERT_EQ(core->CreateTopLevelAccelerationStructure(instances, &tlas), 0);
  sparkium::Core renderer(core.get());
  auto vfs = renderer.GetShadersVFS();
  vfs.WriteFile("probe.hlsl", R"(
#include "compute_contract.hlsli"
struct Hit { float distance; float2 bary; uint instance; uint primitive; };
SP_RESOURCE(RaytracingAccelerationStructure, scene, t0, 0);
SP_RESOURCE(RWByteAddressBuffer, output, u0, 1);
struct Settings { uint enabled; uint mask; uint2 pad; };
SP_RESOURCE(ConstantBuffer<Settings>, settings, b0, 2);
[shader("closesthit")]
void LongMarchOptixClosest(inout Hit hit, BuiltInTriangleIntersectionAttributes attr) {
  hit.distance=RayTCurrent(); hit.bary=attr.barycentrics;
  hit.instance=InstanceID(); hit.primitive=PrimitiveIndex();
}
[shader("miss")] void LongMarchOptixMiss(inout Hit hit) { hit.instance=0xffffffff; }
[shader("raygeneration")] void Main() {
  uint3 id=DispatchRaysIndex();
  if(id.x>=4 || id.y!=0 || id.z!=0) return;
  Hit hit=(Hit)0; hit.instance=0xffffffff;
  RayDesc ray; ray.Origin=float3(id.x*3,0,2); ray.Direction=float3(0,0,-1); ray.TMin=0; ray.TMax=10;
  if(settings.enabled!=0) TraceRay(scene,RAY_FLAG_FORCE_OPAQUE,settings.mask,0,1,0,ray,hit);
  output.Store(id.x*20,hit.instance); output.Store(id.x*20+4,hit.primitive);
  output.Store(id.x*20+8,asuint(hit.distance));
  output.Store(id.x*20+12,asuint(hit.bary.x)); output.Store(id.x*20+16,asuint(hit.bary.y));
})");
  std::unique_ptr<graphics::Shader> shader;
  ASSERT_EQ(core->CreateShader(vfs, "probe.hlsl", "Main", "cs_6_0", {"-DSPARKIUM_OPTIX"}, &shader), 0);
  std::unique_ptr<graphics::ComputeProgram> program;
  core->CreateComputeProgram(shader.get(), &program);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->Finalize();
  auto trace = [&](uint32_t enabled, uint32_t mask) {
    uint32_t params[]{enabled, mask, 0, 0};
    parameters->UploadData(params, sizeof(params));
    std::unique_ptr<graphics::CommandContext> command;
    core->CreateCommandContext(&command);
    command->CmdBindComputeProgram(program.get());
    command->CmdBindResources(0, tlas.get(), graphics::BIND_POINT_COMPUTE);
    command->CmdBindResources(1, std::vector<graphics::Buffer *>{results.get()}, graphics::BIND_POINT_COMPUTE);
    command->CmdBindResources(2, std::vector<graphics::Buffer *>{parameters.get()}, graphics::BIND_POINT_COMPUTE);
    command->CmdDispatch(1, 1, 1);
    core->SubmitCommandContext(command.get());
    std::array<uint32_t, 20> out{};
    results->DownloadData(out.data(), sizeof(out));
    return out;
  };
  auto as_float = [](uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
  };
  auto hit = trace(1, 1);
  EXPECT_EQ(hit[0], 17);
  EXPECT_EQ(hit[1], 0);
  EXPECT_FLOAT_EQ(as_float(hit[2]), 2);
  EXPECT_FLOAT_EQ(as_float(hit[3]), 0.25f);
  EXPECT_FLOAT_EQ(as_float(hit[4]), 0.5f);
  EXPECT_EQ(hit[5], 31);
  EXPECT_FLOAT_EQ(as_float(hit[7]), 1);
  EXPECT_EQ(hit[10], UINT32_MAX);
  EXPECT_EQ(hit[15], UINT32_MAX);
  hit = trace(1, 3);
  EXPECT_EQ(hit[0], 18);
  EXPECT_FLOAT_EQ(as_float(hit[2]), 1.5f);
  instances[1] = blas->MakeInstance(transform(6, 1), 31, 1);
  ASSERT_EQ(tlas->UpdateInstances(instances), 0);
  hit = trace(1, 1);
  EXPECT_EQ(hit[5], UINT32_MAX);
  EXPECT_EQ(hit[10], 31);
  ASSERT_EQ(tlas->UpdateInstances(std::vector<graphics::RayTracingInstance>{}), 0);
  hit = trace(0, 1);
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(hit[i * 5], UINT32_MAX);
  ASSERT_EQ(tlas->UpdateInstances(instances), 0);
  hit = trace(1, 1);
  EXPECT_EQ(hit[0], 17);
  EXPECT_EQ(hit[10], 31);
}
