#include "optix_backend.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <array>
#include <iostream>
#include <mutex>

#if OPTIX_VERSION < 80000
#error "LongMarch requires OptiX SDK 8.0 or newer"
#endif

namespace grassland::graphics::backend {
namespace {
void CheckOptix(OptixResult result, const char *operation, const char *log = "") {
  if (result != OPTIX_SUCCESS)
    throw std::runtime_error(std::string("OptiX ") + operation + ": " + optixGetErrorName(result) + "\n" + log);
}
CUdeviceptr Pointer(const NativeMemory &memory) {
  return reinterpret_cast<CUdeviceptr>(memory.Data());
}
CUdeviceptr BufferPointer(BufferRange range, uint64_t required) {
  auto *buffer = dynamic_cast<NativeBuffer *>(range.buffer);
  if (!buffer || !buffer->memory->IsCUDA())
    throw std::invalid_argument("OptiX acceleration structures require native CUDA device buffers");
  if (range.offset > buffer->Size() || range.size > buffer->Size() - range.offset || required > range.size)
    throw std::out_of_range("OptiX geometry buffer range");
  return Pointer(*buffer->memory) + range.offset;
}
void LogCallback(unsigned int level, const char *tag, const char *message, void *) {
  std::cerr << "OptiX [" << level << "][" << tag << "] " << message << '\n';
}
}  // namespace

OptixDevice::OptixDevice(CUcontext context, bool debug) {
  static std::once_flag initialized;
  std::call_once(initialized, [] { CheckOptix(optixInit(), "initialize driver (check NVIDIA driver installation)"); });
  OptixDeviceContextOptions options{};
  options.logCallbackFunction = LogCallback;
  options.logCallbackLevel = debug ? 4 : 2;
  options.validationMode = debug ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
  CheckOptix(optixDeviceContextCreate(context, &options, &context_), "create device context");
  try {
    uint32_t rtcore_version{};
    CheckOptix(optixDeviceContextGetProperty(context_, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                                            &rtcore_version, sizeof(rtcore_version)), "query hardware traversal");
    if (!rtcore_version)
      throw std::runtime_error("OptiX hardware traversal requires an NVIDIA GPU with RT cores");
  } catch (...) {
    optixDeviceContextDestroy(context_);
    context_ = nullptr;
    throw;
  }
}
OptixDevice::~OptixDevice() {
  if (context_)
    optixDeviceContextDestroy(context_);
}

void OptixAccelerationStructure::Build(const OptixBuildInput &input) {
  OptixAccelBuildOptions options{};
  options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sizes{};
  CheckOptix(optixAccelComputeMemoryUsage(device_->Context(), &options, &input, 1, &sizes), "size acceleration structure");
  NativeMemory scratch(true, sizes.tempSizeInBytes);
  auto output = std::make_unique<NativeMemory>(true, sizes.outputSizeInBytes);
  OptixTraversableHandle handle{};
  CheckOptix(optixAccelBuild(device_->Context(), nullptr, &options, &input, 1, Pointer(scratch), scratch.Size(),
                            Pointer(*output), output->Size(), &handle, nullptr, 0), "build acceleration structure");
  CheckCUDA(cuCtxSynchronize());
  output_ = std::move(output);
  handle_ = handle;
}
OptixAccelerationStructure::OptixAccelerationStructure(OptixDevice *device, BufferRange vertices,
    BufferRange indices, uint32_t vertex_count, uint32_t stride, uint32_t primitive_count, RayTracingGeometryFlag flags)
    : device_(device) {
  if (stride < 12 || stride % 4 || !vertex_count || !primitive_count)
    throw std::invalid_argument("OptiX requires nonempty float3 triangle geometry with a 4-byte aligned stride");
  const auto vertex_ptr = BufferPointer(vertices, uint64_t(vertex_count - 1) * stride + 12);
  const auto index_ptr = BufferPointer(indices, uint64_t(primitive_count) * 12);
  if (vertex_ptr % 4 || index_ptr % 4)
    throw std::invalid_argument("OptiX triangle buffer addresses must be 4-byte aligned");
  uint32_t geometry_flags = OPTIX_GEOMETRY_FLAG_NONE;
  if (flags & RAYTRACING_GEOMETRY_FLAG_OPAQUE)
    geometry_flags |= OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  if (flags & RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION)
    geometry_flags |= OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  auto &triangles = input.triangleArray;
  triangles.vertexBuffers = &vertex_ptr;
  triangles.numVertices = vertex_count;
  triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangles.vertexStrideInBytes = stride;
  triangles.indexBuffer = index_ptr;
  triangles.numIndexTriplets = primitive_count;
  triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangles.indexStrideInBytes = 12;
  triangles.flags = &geometry_flags;
  triangles.numSbtRecords = 1;
  Build(input);
}
OptixAccelerationStructure::OptixAccelerationStructure(OptixDevice *device, const std::vector<RayTracingInstance> &instances)
    : device_(device), top_level_(true) {
  UpdateInstances(instances);
}
int OptixAccelerationStructure::UpdateInstances(const std::vector<RayTracingInstance> &instances) {
  if (!top_level_)
    throw std::invalid_argument("cannot update instances of an OptiX triangle GAS");
  if (instances.size() > UINT32_MAX)
    throw std::overflow_error("OptiX instance count overflow");
  std::vector<OptixInstance> data(instances.size());
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &in = instances[i];
    auto *blas = dynamic_cast<OptixAccelerationStructure *>(in.acceleration_structure);
    if (!blas || blas->Device() != device_ || blas->IsTopLevel())
      throw std::invalid_argument("OptiX IAS requires triangle GAS from the same device");
    if (in.instance_hit_group_offset != 0)
      throw std::invalid_argument("OptiX shared traversal supports one hit group");
    auto &out = data[i];
    std::memcpy(out.transform, in.transform, sizeof(out.transform));
    out.instanceId = in.instance_id;
    out.visibilityMask = in.instance_mask;
    out.traversableHandle = blas->Handle();
    if (in.instance_flags & RAYTRACING_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE)
      out.flags |= OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    if (in.instance_flags & RAYTRACING_INSTANCE_FLAG_TRIANGLE_FLIP_FACING)
      out.flags |= OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING;
    if (in.instance_flags & RAYTRACING_INSTANCE_FLAG_OPAQUE)
      out.flags |= OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
    if (in.instance_flags & RAYTRACING_INSTANCE_FLAG_NO_OPAQUE)
      out.flags |= OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT;
  }
  if (data.size() == previous_instances_.size() &&
      (data.empty() || std::memcmp(data.data(), previous_instances_.data(), data.size() * sizeof(OptixInstance)) == 0))
    return 0;
  if (data.empty()) {
    handle_ = 0;  // Shared OptiX traversal skips optixTrace for an empty scene.
    output_.reset();
    instances_.reset();
  } else {
    auto next = std::make_unique<NativeMemory>(true, data.size() * sizeof(OptixInstance));
    next->Upload(data.data(), next->Size());
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = Pointer(*next);
    input.instanceArray.numInstances = static_cast<unsigned>(data.size());
    Build(input);
    instances_ = std::move(next);
  }
  previous_instances_ = std::move(data);
  return 0;
}

OptixLaunch::OptixLaunch(OptixDevice *device, const std::string &ptx, const std::string &entry, size_t params_size) {
  try {
    OptixModuleCompileOptions module_options{};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions options{};
    options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    // Slang serializes the hit payload (float, float2, uint, uint) into registers;
    // reserve eight to cover native float2 padding without imposing a C++ layout.
    options.numPayloadValues = 8;
    options.numAttributeValues = 2;
    options.pipelineLaunchParamsVariableName = "SLANG_globalParams";
    options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[16384]{};
    size_t log_size = sizeof(log);
    auto status = optixModuleCreate(device->Context(), &module_options, &options, ptx.data(), ptx.size(), log,
                                     &log_size, &module_);
    CheckOptix(status, "compile PTX module", log);
    std::string raygen = "__raygen__" + entry;
    OptixProgramGroupDesc descriptions[3]{};
    descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptions[0].raygen.module = module_;
    descriptions[0].raygen.entryFunctionName = raygen.c_str();
    descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptions[1].miss.module = module_;
    descriptions[1].miss.entryFunctionName = "__miss__LongMarchOptixMiss";
    descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptions[2].hitgroup.moduleCH = module_;
    descriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__LongMarchOptixClosest";
    OptixProgramGroupOptions group_options{};
    log_size = sizeof(log);
    status = optixProgramGroupCreate(device->Context(), descriptions, 3, &group_options, log, &log_size, groups_);
    CheckOptix(status, "create shader groups", log);
    OptixPipelineLinkOptions link{};
    link.maxTraceDepth = 1;  // Iterative path tracing; hit/miss programs never trace recursively.
    log_size = sizeof(log);
    status = optixPipelineCreate(device->Context(), &options, &link, groups_, 3, log, &log_size, &pipeline_);
    CheckOptix(status, "link pipeline", log);
    OptixStackSizes stack{};
    for (auto group : groups_)
      CheckOptix(optixUtilAccumulateStackSizes(group, &stack, pipeline_), "query stack sizes");
    uint32_t traversal{}, state{}, continuation{};
    CheckOptix(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation), "compute stack sizes");
    CheckOptix(optixPipelineSetStackSize(pipeline_, traversal, state, continuation, 2), "set pipeline stack");
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
      char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    std::array<Record, 3> records{};
    for (size_t i = 0; i < records.size(); ++i)
      CheckOptix(optixSbtRecordPackHeader(groups_[i], &records[i]), "pack shader binding table");
    records_ = std::make_unique<NativeMemory>(true, sizeof(records));
    records_->Upload(records.data(), sizeof(records));
    sbt_.raygenRecord = Pointer(*records_);
    sbt_.missRecordBase = Pointer(*records_) + sizeof(Record);
    sbt_.missRecordStrideInBytes = sizeof(Record);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = Pointer(*records_) + 2 * sizeof(Record);
    sbt_.hitgroupRecordStrideInBytes = sizeof(Record);
    sbt_.hitgroupRecordCount = 1;
    params_ = std::make_unique<NativeMemory>(true, params_size);
  } catch (...) {
    Destroy();
    throw;
  }
}
void OptixLaunch::Destroy() noexcept {
  if (pipeline_)
    optixPipelineDestroy(pipeline_);
  for (auto group : groups_)
    if (group)
      optixProgramGroupDestroy(group);
  if (module_)
    optixModuleDestroy(module_);
}
OptixLaunch::~OptixLaunch() {
  Destroy();
}
void OptixLaunch::Dispatch(const void *data, size_t size, uint32_t x, uint32_t y, uint32_t z) {
  if (size != params_->Size())
    throw std::runtime_error("OptiX launch parameter ABI mismatch");
  params_->Upload(data, size);
  CheckOptix(optixLaunch(pipeline_, nullptr, Pointer(*params_), size, &sbt_, x, y, z), "launch");
  CheckCUDA(cuCtxSynchronize());
}
}  // namespace grassland::graphics::backend
