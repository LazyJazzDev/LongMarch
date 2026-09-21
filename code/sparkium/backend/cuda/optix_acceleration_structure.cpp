#include "sparkium/backend/cuda/optix_acceleration_structure.h"

#include "sparkium/backend/cuda/cuda_buffer.h"
#include "sparkium/backend/cuda/driver_util.h"
#include "sparkium/backend/cuda/optix_util.h"

namespace sparkium::backend {
using namespace cuda;

namespace {
CUdeviceptr BufferPointer(BufferRange range, uint64_t required) {
  auto *buffer = dynamic_cast<CudaBuffer *>(range.buffer);
  if (!buffer)
    throw std::invalid_argument("OptiX acceleration structures require compute CUDA device buffers");
  if (range.offset > buffer->Size() || range.size > buffer->Size() - range.offset || required > range.size)
    throw std::out_of_range("OptiX geometry buffer range");
  return Pointer(*buffer->memory) + range.offset;
}
}  // namespace

void OptixAccelerationStructure::Build(const OptixBuildInput &input) {
  OptixAccelBuildOptions options{};
  options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sizes{};
  CheckOptix(optixAccelComputeMemoryUsage(device_->Context(), &options, &input, 1, &sizes),
             "size acceleration structure");
  CudaMemory scratch(sizes.tempSizeInBytes);
  auto output = std::make_unique<CudaMemory>(sizes.outputSizeInBytes);
  OptixTraversableHandle handle{};
  CheckOptix(optixAccelBuild(device_->Context(), nullptr, &options, &input, 1, Pointer(scratch), scratch.Size(),
                             Pointer(*output), output->Size(), &handle, nullptr, 0),
             "build acceleration structure");
  CheckCUDA(cuCtxSynchronize());
  output_ = std::move(output);
  handle_ = handle;
}

OptixAccelerationStructure::OptixAccelerationStructure(OptixDevice *device,
                                                       BufferRange vertices,
                                                       BufferRange indices,
                                                       uint32_t vertex_count,
                                                       uint32_t stride,
                                                       uint32_t primitive_count,
                                                       RayTracingGeometryFlag flags)
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

OptixAccelerationStructure::OptixAccelerationStructure(OptixDevice *device,
                                                       const std::vector<RayTracingInstance> &instances)
    : device_(device),
      top_level_(true) {
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
    auto next = std::make_unique<CudaMemory>(data.size() * sizeof(OptixInstance));
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

}  // namespace sparkium::backend
