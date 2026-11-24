#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_compute_program.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi::backend {
// Offline preparation calls this with the target cache settings to include the
// internal mip shader and its exact pipeline layout without recording GPU work.
ObjectPtr<IComputeProgram> CreateMipGenerationProgram(IDevice *device);

// Caller validates ownership, capabilities and recording scope. Cached pipeline
// objects belong to this context; captured commands retain them through completion.
bool GenerateMipChain(ICommandContext *commands,
                      IDevice *device,
                      IImage *image,
                      uint32_t base_mip,
                      uint32_t mip_count,
                      ObjectPtr<IComputeProgram> &program,
                      ObjectPtr<ISampler> &sampler);
}  // namespace grassland::rhi::backend
