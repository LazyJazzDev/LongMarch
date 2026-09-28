#include "grassland/graphics/backend/d3d12/d3d12_util.h"

#include "grassland/graphics/backend/d3d12/d3d12_acceleration_structure.h"

namespace grassland::graphics::backend {
Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> CreateNativeDescriptorHeap(ID3D12Device *device,
                                                                        D3D12_DESCRIPTOR_HEAP_TYPE type,
                                                                        uint32_t count) {
  D3D12_DESCRIPTOR_HEAP_DESC desc{};
  desc.Type = type;
  desc.NumDescriptors = std::max(count, 1u);
  if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  }
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
  d3d12::ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.GetAddressOf())),
                       "Failed to create descriptor heap");
  return heap;
}

Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateNativeRootSignature(
    ID3D12Device *device,
    const CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC &desc) {
  D3D12_FEATURE_DATA_ROOT_SIGNATURE feature_data{};
  feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
  if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature_data, sizeof(feature_data)))) {
    feature_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
  }

  Microsoft::WRL::ComPtr<ID3DBlob> signature;
  Microsoft::WRL::ComPtr<ID3DBlob> error;
  HRESULT result = D3DX12SerializeVersionedRootSignature(&desc, feature_data.HighestVersion, &signature, &error);
  if (FAILED(result) && error) {
    d3d12::SetErrorMessage("Failed to serialize root signature: {}",
                           static_cast<const char *>(error->GetBufferPointer()));
  }
  d3d12::ThrowIfFailed(result, "Failed to serialize root signature");

  Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature;
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
  desc.AccelerationStructure = dynamic_cast<D3D12AccelerationStructure *>(instance.acceleration_structure)
                                   ->Handle()
                                   ->Handle()
                                   ->GetGPUVirtualAddress();
  return desc;
}

D3D12ResourceBinding::D3D12ResourceBinding() : buffer(nullptr), image(nullptr) {
}

D3D12ResourceBinding::D3D12ResourceBinding(D3D12Buffer *buffer) : buffer(buffer), image(nullptr) {
}

D3D12ResourceBinding::D3D12ResourceBinding(D3D12Image *image) : buffer(nullptr), image(image) {
}

}  // namespace grassland::graphics::backend
