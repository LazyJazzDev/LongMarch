#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12AccelerationStructure;
class ID3D12AllocatedBuffer;
class ID3D12BindingVersion;
class ID3D12BindingGroup;
class ID3D12Buffer;
class ID3D12ComputeProgram;
class ID3D12Device;
class ID3D12Image;
class ID3D12Program;
class ID3D12ProgramBase;
class ID3D12Queue;
class ID3D12RayTracingProgram;
class ID3D12Sampler;
class ID3D12SwapChainImage;
class ID3D12TemporalBufferLease;
class ID3D12Timeline;
class ID3D12TopLevelAccelerationStructure;
class ID3D12Window;

D3D12_PRIMITIVE_TOPOLOGY NativePrimitiveTopology(PrimitiveTopology topology);

D3D12_BLEND NativeBlendFactor(BlendFactor factor, bool alpha = false);
D3D12_BLEND_OP NativeBlendOp(BlendOp op);
bool SupportsImageStorage(::ID3D12Device *device, ImageFormat format);
DXGI_FORMAT NativeImageFormat(ImageFormat format);
using Microsoft::WRL::ComPtr;
bool Check(HRESULT result, const char *operation);
}  // namespace grassland::rhi::backend::d3d12
