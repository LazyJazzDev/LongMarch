#include "optix_launch.h"

#include <optix_stack_size.h>

#include <array>

#include "optix_util.h"

namespace grassland::graphics::backend {

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
