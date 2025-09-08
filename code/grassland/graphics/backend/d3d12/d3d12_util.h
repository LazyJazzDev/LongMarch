#pragma once
#include "grassland/d3d12/direct3d12.h"
#include "grassland/graphics/interface.h"

namespace CD::graphics::backend {

DXGI_FORMAT ImageFormatToDXGIFormat(ImageFormat format);

DXGI_FORMAT InputTypeToDXGIFormat(InputType type);

D3D12_DESCRIPTOR_RANGE_TYPE ResourceTypeToD3D12DescriptorRangeType(ResourceType type);

D3D12_CULL_MODE CullModeToD3D12CullMode(CullMode mode);

D3D12_FILTER FilterModeToD3D12Filter(FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter);

D3D12_TEXTURE_ADDRESS_MODE AddressModeToD3D12AddressMode(AddressMode mode);

D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopologyToD3D12PrimitiveTopology(PrimitiveTopology topology);

D3D12_BLEND BlendFactorToD3D12Blend(BlendFactor factor);

D3D12_BLEND_OP BlendOpToD3D12BlendOp(BlendOp op);

D3D12_RENDER_TARGET_BLEND_DESC BlendStateToD3D12RenderTargetBlendDesc(const BlendState &state);

D3D12_RAYTRACING_INSTANCE_DESC RayTracingInstanceToD3D12RayTracingInstanceDesc(const RayTracingInstance &instance);

class D3D12Core;
class D3D12Buffer;
class D3D12Image;
class D3D12Sampler;
class D3D12Shader;
class D3D12ProgramBase;
class D3D12Program;
class D3D12ComputeProgram;
class D3D12CommandContext;
class D3D12Window;
class D3D12AccelerationStructure;
class D3D12RayTracingProgram;

struct D3D12BufferRange;

struct D3D12ResourceBinding {
  D3D12ResourceBinding();

  D3D12ResourceBinding(D3D12Buffer *buffer);

  D3D12ResourceBinding(D3D12Image *image);

  D3D12Buffer *buffer;
  D3D12Image *image;
};

#if defined(LONGMARCH_CUDA_RUNTIME)
class D3D12CUDABuffer;
#endif

}  // namespace CD::graphics::backend
