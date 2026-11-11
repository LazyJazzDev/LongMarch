#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::metal {
class IMetalAccelerationStructure;
class IMetalAllocatedBuffer;
class IMetalBindingVersion;
class IMetalBuffer;
class IMetalComputeProgram;
class IMetalCore;
class IMetalDevice;
class IMetalImage;
class IMetalProgram;
class IMetalProgramBase;
class IMetalQueue;
class IMetalSampler;
class IMetalSwapChainImage;
class IMetalTemporalBufferLease;
class IMetalTimeline;
class IMetalTopLevelAccelerationStructure;
class IMetalWindow;

MTL::PrimitiveType NativePrimitiveTopology(PrimitiveTopology topology);

MTL::BlendFactor NativeBlendFactor(BlendFactor factor, bool alpha = false);
MTL::BlendOperation NativeBlendOp(BlendOp op);
bool SupportsImageStorage(ImageFormat format);
MTL::PixelFormat NativeImageFormat(ImageFormat format);
}  // namespace grassland::rhi::backend::metal
