#pragma once
#define NOMINMAX

#include <d3d12.h>
#include <d3dcompiler.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"
#include "glm/glm.hpp"
#include "grassland/graphics/graphics_util.h"
#include "grassland/util/util.h"

namespace grassland::d3d12 {

using Microsoft::WRL::ComPtr;

void ThrowError(const std::string &message);

template <class... Args>
void ThrowError(const std::string &message, Args &&...args) {
  ThrowError(fmt::format(message, std::forward<Args>(args)...));
}

void ThrowIfFailed(HRESULT hr, const std::string &message);

template <class... Args>
void ThrowIfFailed(HRESULT hr, const std::string &message, Args &&...args) {
  ThrowIfFailed(hr, fmt::format(message, std::forward<Args>(args)...));
}

void Warning(const std::string &message);

template <class... Args>
void Warning(const std::string &message, Args &&...args) {
  Warning(fmt::format(message, std::forward<Args>(args)...));
}

void SetErrorMessage(const std::string &message);

template <class... Args>
void SetErrorMessage(const std::string &message, Args &&...args) {
  SetErrorMessage(fmt::format(message, std::forward<Args>(args)...));
}

std::string GetErrorMessage();

std::string HRESULTToString(HRESULT hr);

size_t SizeByFormat(DXGI_FORMAT format);

size_t SizeAlignTo(size_t size, size_t alignment);

#define RETURN_IF_FAILED_HR(cmd, ...)                                                               \
  do {                                                                                              \
    HRESULT res = cmd;                                                                              \
    if (FAILED(res)) {                                                                              \
      ::grassland::d3d12::SetErrorMessage(__VA_ARGS__);                                             \
      ::grassland::d3d12::SetErrorMessage("HRESULT: {}", ::grassland::d3d12::HRESULTToString(res)); \
      return res;                                                                                   \
    }                                                                                               \
                                                                                                    \
  } while (false)

struct DeviceFeatureRequirement;

class DXGIFactory;
class Adapter;
class Device;
class SwapChain;
class DescriptorHeap;
class RootSignature;
class CommandQueue;
class CommandAllocator;
class CommandList;
class Buffer;
class Image;
class Fence;
class ShaderModule;
class PipelineState;
class AccelerationStructure;
class RayTracingPipeline;
class ShaderTable;
struct HitGroup;

#ifdef NDEBUG
constexpr bool kDefaultEnableDebugLayer = false;
#else
constexpr bool kDefaultEnableDebugLayer = true;
#endif

bool IsDepthFormat(DXGI_FORMAT format);

HRESULT CreateBuffer(ID3D12Device *device,
                     size_t size,
                     D3D12_HEAP_TYPE heap_type,
                     D3D12_HEAP_FLAGS heap_flags,
                     D3D12_RESOURCE_STATES resource_state,
                     D3D12_RESOURCE_FLAGS resource_flags,
                     ComPtr<ID3D12Resource> &buffer);

HRESULT CreateBuffer(ID3D12Device *device,
                     size_t size,
                     D3D12_HEAP_TYPE heap_type,
                     D3D12_RESOURCE_STATES resource_state,
                     D3D12_RESOURCE_FLAGS resource_flags,
                     ComPtr<ID3D12Resource> &buffer);


D3D12_RESOURCE_STATES HeapTypeDefaultResourceState(D3D12_HEAP_TYPE heap_type);

using graphics::CompiledShaderBlob;

}  // namespace grassland::d3d12
