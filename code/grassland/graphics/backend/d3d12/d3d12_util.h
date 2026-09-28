#pragma once
#include "grassland/graphics/backend/d3d12/helper/direct3d12.h"
#include "grassland/graphics/interface.h"

namespace grassland::graphics::backend {

Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateNativeRootSignature(
    ID3D12Device *device,
    const CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC &desc);

Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> CreateNativeDescriptorHeap(ID3D12Device *device,
                                                                        D3D12_DESCRIPTOR_HEAP_TYPE type,
                                                                        uint32_t count);

Microsoft::WRL::ComPtr<ID3D12Resource> CreateNativeBuffer(ID3D12Device *device,
                                                          size_t size,
                                                          D3D12_HEAP_TYPE heap_type,
                                                          D3D12_HEAP_FLAGS heap_flags,
                                                          D3D12_RESOURCE_STATES resource_state,
                                                          D3D12_RESOURCE_FLAGS resource_flags);

Microsoft::WRL::ComPtr<ID3D12Resource> CreateNativeBuffer(ID3D12Device *device, size_t size, D3D12_HEAP_TYPE heap_type);

Microsoft::WRL::ComPtr<ID3D12Resource> CreateNativeImage(ID3D12Device *device,
                                                         size_t width,
                                                         size_t height,
                                                         DXGI_FORMAT format);

Microsoft::WRL::ComPtr<IDXGISwapChain3> CreateNativeSwapChain(IDXGIFactory4 *factory,
                                                              ID3D12CommandQueue *queue,
                                                              HWND hwnd,
                                                              uint32_t buffer_count,
                                                              DXGI_FORMAT format);

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

}  // namespace grassland::graphics::backend
