#include "grassland/graphics/backend/d3d12/d3d12_util.h"

#include "grassland/graphics/backend/d3d12/d3d12_acceleration_structure.h"

namespace grassland::graphics::backend {
ComPtr<ID3D12Resource> CreateNativeBuffer(ID3D12Device *device,
                                          size_t size,
                                          D3D12_HEAP_TYPE heap_type,
                                          D3D12_HEAP_FLAGS heap_flags,
                                          D3D12_RESOURCE_STATES resource_state,
                                          D3D12_RESOURCE_FLAGS resource_flags) {
  ComPtr<ID3D12Resource> buffer;
  d3d12::ThrowIfFailed(d3d12::CreateBuffer(device, size, heap_type, heap_flags, resource_state, resource_flags, buffer),
                       "Failed to create buffer");
  return buffer;
}

ComPtr<ID3D12Resource> CreateNativeBuffer(ID3D12Device *device, size_t size, D3D12_HEAP_TYPE heap_type) {
  return CreateNativeBuffer(
      device, size, heap_type, D3D12_HEAP_FLAG_NONE, d3d12::HeapTypeDefaultResourceState(heap_type),
      heap_type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
}

ComPtr<ID3D12Resource> CreateNativeImage(ID3D12Device *device, size_t width, size_t height, DXGI_FORMAT format) {
  const auto flags = d3d12::IsDepthFormat(format)
                         ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                         : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
  auto desc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1, 1, 0, flags);
  D3D12_CLEAR_VALUE clear_value{};
  clear_value.Format = format;
  if (d3d12::IsDepthFormat(format)) {
    clear_value.DepthStencil.Depth = 1.0f;
  } else {
    clear_value.Color[3] = 1.0f;
  }
  const CD3DX12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE_DEFAULT);
  ComPtr<ID3D12Resource> image;
  d3d12::ThrowIfFailed(
      device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      &clear_value, IID_PPV_ARGS(image.GetAddressOf())),
      "Failed to create image");
  return image;
}

ComPtr<IDXGISwapChain3> CreateNativeSwapChain(IDXGIFactory4 *factory,
                                              ID3D12CommandQueue *queue,
                                              HWND hwnd,
                                              uint32_t buffer_count,
                                              DXGI_FORMAT format) {
  RECT rect{};
  GetClientRect(hwnd, &rect);
  DXGI_SWAP_CHAIN_DESC1 desc{};
  desc.BufferCount = buffer_count;
  desc.Width = rect.right - rect.left;
  desc.Height = rect.bottom - rect.top;
  desc.Format = format;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.SampleDesc.Count = 1;

  ComPtr<IDXGISwapChain1> swap_chain1;
  d3d12::ThrowIfFailed(
      factory->CreateSwapChainForHwnd(queue, hwnd, &desc, nullptr, nullptr, swap_chain1.GetAddressOf()),
      "Failed to create swap chain");
  ComPtr<IDXGISwapChain3> swap_chain;
  d3d12::ThrowIfFailed(swap_chain1.As(&swap_chain), "Failed to query swap chain 3");
  if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
    constexpr auto color_space = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
    UINT support = 0;
    if (SUCCEEDED(swap_chain->CheckColorSpaceSupport(color_space, &support)) &&
        (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
      d3d12::ThrowIfFailed(swap_chain->SetColorSpace1(color_space), "Failed to set HDR color space");
    }
  }
  return swap_chain;
}

std::vector<ComPtr<IDXGIAdapter1>> EnumerateNativeAdapters(IDXGIFactory4 *factory) {
  std::vector<ComPtr<IDXGIAdapter1>> adapters;
  for (UINT index = 0;; ++index) {
    ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(index, adapter.GetAddressOf()) == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      adapters.push_back(std::move(adapter));
    }
  }
  return adapters;
}

std::string NativeAdapterName(IDXGIAdapter1 *adapter) {
  DXGI_ADAPTER_DESC1 desc{};
  adapter->GetDesc1(&desc);
  return WStringToString(desc.Description);
}

bool NativeAdapterSupportsRayTracing(IDXGIAdapter1 *adapter) {
  ComPtr<ID3D12Device5> device;
  if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())))) {
    return false;
  }
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
  return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options))) &&
         options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
}

uint64_t NativeAdapterScore(IDXGIAdapter1 *adapter) {
  DXGI_ADAPTER_DESC1 desc{};
  adapter->GetDesc1(&desc);
  uint64_t score = desc.DedicatedVideoMemory / 1024 / 1024;
  if (NativeAdapterSupportsRayTracing(adapter)) {
    score += 100000;
  }
  return score;
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int NativeAdapterCUDADeviceIndex(IDXGIAdapter1 *adapter) {
  DXGI_ADAPTER_DESC1 desc{};
  adapter->GetDesc1(&desc);
  int count = 0;
  cudaGetDeviceCount(&count);
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, index);
    if (std::memcmp(&properties.luid, &desc.AdapterLuid, sizeof(LUID)) == 0) {
      return index;
    }
  }
  return -1;
}
#endif

ComPtr<ID3D12DescriptorHeap> CreateNativeDescriptorHeap(ID3D12Device *device,
                                                        D3D12_DESCRIPTOR_HEAP_TYPE type,
                                                        uint32_t count) {
  D3D12_DESCRIPTOR_HEAP_DESC desc{};
  desc.Type = type;
  desc.NumDescriptors = std::max(count, 1u);
  if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  }
  ComPtr<ID3D12DescriptorHeap> heap;
  d3d12::ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.GetAddressOf())),
                       "Failed to create descriptor heap");
  return heap;
}

ComPtr<ID3D12RootSignature> CreateNativeRootSignature(ID3D12Device *device,
                                                      const CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC &desc) {
  D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data{};
  feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
  if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data)))) {
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
  }

  ComPtr<ID3DBlob> signature;
  ComPtr<ID3DBlob> error;
  HRESULT result = D3DX12SerializeVersionedRootSignature(&desc, feature_data.HighestVersion, &signature, &error);
  if (FAILED(result) && error) {
    d3d12::SetErrorMessage("Failed to serialize root signature: {}",
                           static_cast<const char *>(error->GetBufferPointer()));
  }
  d3d12::ThrowIfFailed(result, "Failed to serialize root signature");

  ComPtr<ID3D12RootSignature> root_signature;
  d3d12::ThrowIfFailed(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                   IID_PPV_ARGS(root_signature.GetAddressOf())),
                       "Failed to create root signature");
  return root_signature;
}

DXGI_FORMAT ImageFormatToDXGIFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return DXGI_FORMAT_R32G32B32_FLOAT;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return DXGI_FORMAT_R32G32_FLOAT;
    case IMAGE_FORMAT_R32_SFLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case IMAGE_FORMAT_D32_SFLOAT:
      return DXGI_FORMAT_D32_FLOAT;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case IMAGE_FORMAT_R32_UINT:
      return DXGI_FORMAT_R32_UINT;
    case IMAGE_FORMAT_R32_SINT:
      return DXGI_FORMAT_R32_SINT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT InputTypeToDXGIFormat(InputType type) {
  switch (type) {
    case INPUT_TYPE_INT:
      return DXGI_FORMAT_R32_SINT;
    case INPUT_TYPE_UINT:
      return DXGI_FORMAT_R32_UINT;
    case INPUT_TYPE_FLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case INPUT_TYPE_INT2:
      return DXGI_FORMAT_R32G32_SINT;
    case INPUT_TYPE_UINT2:
      return DXGI_FORMAT_R32G32_UINT;
    case INPUT_TYPE_FLOAT2:
      return DXGI_FORMAT_R32G32_FLOAT;
    case INPUT_TYPE_INT3:
      return DXGI_FORMAT_R32G32B32_SINT;
    case INPUT_TYPE_UINT3:
      return DXGI_FORMAT_R32G32B32_UINT;
    case INPUT_TYPE_FLOAT3:
      return DXGI_FORMAT_R32G32B32_FLOAT;
    case INPUT_TYPE_INT4:
      return DXGI_FORMAT_R32G32B32A32_SINT;
    case INPUT_TYPE_UINT4:
      return DXGI_FORMAT_R32G32B32A32_UINT;
    case INPUT_TYPE_FLOAT4:
      return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

D3D12_DESCRIPTOR_RANGE_TYPE ResourceTypeToD3D12DescriptorRangeType(ResourceType type) {
  switch (type) {
    case RESOURCE_TYPE_UNIFORM_BUFFER:
      return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case RESOURCE_TYPE_STORAGE_BUFFER:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case RESOURCE_TYPE_IMAGE:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case RESOURCE_TYPE_WRITABLE_IMAGE:
      return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    case RESOURCE_TYPE_SAMPLER:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    case RESOURCE_TYPE_ACCELERATION_STRUCTURE:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER:
      return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    default:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  }
}

D3D12_CULL_MODE CullModeToD3D12CullMode(CullMode mode) {
  switch (mode) {
    case CULL_MODE_NONE:
      return D3D12_CULL_MODE_NONE;
    case CULL_MODE_BACK:
      return D3D12_CULL_MODE_BACK;
    case CULL_MODE_FRONT:
      return D3D12_CULL_MODE_FRONT;
    default:
      return D3D12_CULL_MODE_NONE;
  }
}

D3D12_FILTER FilterModeToD3D12Filter(FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter) {
  if (min_filter == FILTER_MODE_NEAREST && mag_filter == FILTER_MODE_NEAREST && mip_filter == FILTER_MODE_NEAREST) {
    return D3D12_FILTER_MIN_MAG_MIP_POINT;
  } else if (min_filter == FILTER_MODE_NEAREST && mag_filter == FILTER_MODE_NEAREST &&
             mip_filter == FILTER_MODE_LINEAR) {
    return D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR;
  } else if (min_filter == FILTER_MODE_NEAREST && mag_filter == FILTER_MODE_LINEAR &&
             mip_filter == FILTER_MODE_NEAREST) {
    return D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
  } else if (min_filter == FILTER_MODE_NEAREST && mag_filter == FILTER_MODE_LINEAR &&
             mip_filter == FILTER_MODE_LINEAR) {
    return D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR;
  } else if (min_filter == FILTER_MODE_LINEAR && mag_filter == FILTER_MODE_NEAREST &&
             mip_filter == FILTER_MODE_NEAREST) {
    return D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
  } else if (min_filter == FILTER_MODE_LINEAR && mag_filter == FILTER_MODE_NEAREST &&
             mip_filter == FILTER_MODE_LINEAR) {
    return D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
  } else if (min_filter == FILTER_MODE_LINEAR && mag_filter == FILTER_MODE_LINEAR &&
             mip_filter == FILTER_MODE_NEAREST) {
    return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
  } else if (min_filter == FILTER_MODE_LINEAR && mag_filter == FILTER_MODE_LINEAR && mip_filter == FILTER_MODE_LINEAR) {
    return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  }
  return D3D12_FILTER_MIN_MAG_MIP_POINT;
}

D3D12_TEXTURE_ADDRESS_MODE AddressModeToD3D12AddressMode(AddressMode mode) {
  switch (mode) {
    case ADDRESS_MODE_REPEAT:
      return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    case ADDRESS_MODE_MIRRORED_REPEAT:
      return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    case ADDRESS_MODE_CLAMP_TO_EDGE:
      return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    case ADDRESS_MODE_CLAMP_TO_BORDER:
      return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
  }
  return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
}

D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopologyToD3D12PrimitiveTopology(PrimitiveTopology topology) {
  switch (topology) {
    case PRIMITIVE_TOPOLOGY_LINE_LIST:
      return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case PRIMITIVE_TOPOLOGY_POINT_LIST:
      return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    case PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
  }
  return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
}

D3D12_BLEND BlendFactorToD3D12Blend(BlendFactor factor) {
  switch (factor) {
    case BLEND_FACTOR_ZERO:
      return D3D12_BLEND_ZERO;
    case BLEND_FACTOR_ONE:
      return D3D12_BLEND_ONE;
    case BLEND_FACTOR_SRC_COLOR:
      return D3D12_BLEND_SRC_COLOR;
    case BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
      return D3D12_BLEND_INV_SRC_COLOR;
    case BLEND_FACTOR_DST_COLOR:
      return D3D12_BLEND_DEST_COLOR;
    case BLEND_FACTOR_ONE_MINUS_DST_COLOR:
      return D3D12_BLEND_INV_DEST_COLOR;
    case BLEND_FACTOR_SRC_ALPHA:
      return D3D12_BLEND_SRC_ALPHA;
    case BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
      return D3D12_BLEND_INV_SRC_ALPHA;
    case BLEND_FACTOR_DST_ALPHA:
      return D3D12_BLEND_DEST_ALPHA;
    case BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
      return D3D12_BLEND_INV_DEST_ALPHA;
  }
  return D3D12_BLEND_ZERO;
}

D3D12_BLEND_OP BlendOpToD3D12BlendOp(BlendOp op) {
  switch (op) {
    case BLEND_OP_ADD:
      return D3D12_BLEND_OP_ADD;
    case BLEND_OP_SUBTRACT:
      return D3D12_BLEND_OP_SUBTRACT;
    case BLEND_OP_REVERSE_SUBTRACT:
      return D3D12_BLEND_OP_REV_SUBTRACT;
    case BLEND_OP_MIN:
      return D3D12_BLEND_OP_MIN;
    case BLEND_OP_MAX:
      return D3D12_BLEND_OP_MAX;
  }
  return D3D12_BLEND_OP_ADD;
}

D3D12_RENDER_TARGET_BLEND_DESC BlendStateToD3D12RenderTargetBlendDesc(const BlendState &state) {
  D3D12_RENDER_TARGET_BLEND_DESC desc{};
  desc.BlendEnable = state.blend_enable;
  desc.SrcBlend = BlendFactorToD3D12Blend(state.src_color);
  desc.DestBlend = BlendFactorToD3D12Blend(state.dst_color);
  desc.BlendOp = BlendOpToD3D12BlendOp(state.color_op);
  desc.SrcBlendAlpha = BlendFactorToD3D12Blend(state.src_alpha);
  desc.DestBlendAlpha = BlendFactorToD3D12Blend(state.dst_alpha);
  desc.BlendOpAlpha = BlendOpToD3D12BlendOp(state.alpha_op);
  desc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN |
                               D3D12_COLOR_WRITE_ENABLE_BLUE | D3D12_COLOR_WRITE_ENABLE_ALPHA;
  desc.LogicOpEnable = FALSE;
  desc.LogicOp = D3D12_LOGIC_OP_NOOP;
  return desc;
}

D3D12_RAYTRACING_INSTANCE_DESC RayTracingInstanceToD3D12RayTracingInstanceDesc(const RayTracingInstance &instance) {
  D3D12_RAYTRACING_INSTANCE_DESC desc{};
  std::memcpy(desc.Transform, instance.transform, sizeof(instance.transform));
  desc.InstanceID = instance.instance_id;
  desc.InstanceMask = instance.instance_mask;
  desc.InstanceContributionToHitGroupIndex = instance.instance_hit_group_offset;
  desc.Flags = instance.instance_flags;
  desc.AccelerationStructure =
      dynamic_cast<D3D12AccelerationStructure *>(instance.acceleration_structure)->Handle()->GetGPUVirtualAddress();
  return desc;
}

D3D12ResourceBinding::D3D12ResourceBinding() : buffer(nullptr), image(nullptr) {
}

D3D12ResourceBinding::D3D12ResourceBinding(D3D12Buffer *buffer) : buffer(buffer), image(nullptr) {
}

D3D12ResourceBinding::D3D12ResourceBinding(D3D12Image *image) : buffer(nullptr), image(image) {
}

}  // namespace grassland::graphics::backend
