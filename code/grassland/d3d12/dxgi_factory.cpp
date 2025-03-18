#include "grassland/d3d12/dxgi_factory.h"

#include "grassland/d3d12/adapter.h"
#include "grassland/d3d12/command_queue.h"
#include "grassland/d3d12/device.h"
#include "grassland/d3d12/device_feature_requirement.h"
#include "grassland/d3d12/swap_chain.h"

namespace grassland::d3d12 {

void ThrowError(const std::string &message) {
  throw std::runtime_error(message);
}

void ThrowIfFailed(HRESULT hr, const std::string &message) {
  if (FAILED(hr)) {
    ThrowError(message);
  }
}

void Warning(const std::string &message) {
  LogWarning("[D3D12] " + message);
}

namespace {
std::string error_message;
}  // namespace

void SetErrorMessage(const std::string &message) {
  LogError("[D3D12] " + message);
  error_message = message;
}

std::string GetErrorMessage() {
  return error_message;
}

DXGIFactoryCreateHint::DXGIFactoryCreateHint(bool enable_debug) : enable_debug(enable_debug) {
}

DXGIFactory::DXGIFactory(DXGIFactoryCreateHint hint, const ComPtr<IDXGIFactory4> &factory)
    : hint_(hint), factory_(factory) {
}

std::vector<Adapter> DXGIFactory::EnumerateAdapters() const {
  std::vector<Adapter> adapters;
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0; factory_->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      continue;
    }
    adapters.emplace_back(adapter);
  }
  return adapters;
}

HRESULT DXGIFactory::CreateDevice(const DeviceFeatureRequirement &device_feature_requirement,
                                  int device_index,
                                  double_ptr<Device> pp_device) {
  auto adapters = EnumerateAdapters();

  if (device_index < 0) {
    uint64_t max_score = 0;
    for (int i = 0; i < adapters.size(); i++) {
      if (!adapters[i].CheckFeatureSupport(device_feature_requirement)) {
        continue;
      }

      uint64_t score = adapters[i].Evaluate();
      if (device_index < 0 || score > max_score) {
        max_score = score;
        device_index = i;
      }
    }
  }

  if (device_index < 0 || device_index >= adapters.size()) {
    SetErrorMessage("no suitable physical device found.");
    return E_FAIL;
  }

  ComPtr<ID3D12Device> device;

  const D3D_FEATURE_LEVEL min_feature_level = D3D_FEATURE_LEVEL_11_0;

  HRESULT hr = D3D12CreateDevice(adapters[device_index].Handle(), min_feature_level, IID_PPV_ARGS(&device));

  if (FAILED(hr)) {
    SetErrorMessage("failed to create device.");
    return hr;
  }

  if (hint_.enable_debug) {
    // Configure debug device (if active).
    ComPtr<ID3D12InfoQueue> d3dInfoQueue;
    if (SUCCEEDED(device.As(&d3dInfoQueue))) {
      d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
      d3dInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
      D3D12_MESSAGE_ID hide[] = {D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE, D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE};
      D3D12_INFO_QUEUE_FILTER filter = {};
      filter.DenyList.NumIDs = _countof(hide);
      filter.DenyList.pIDList = hide;
      d3dInfoQueue->AddStorageFilterEntries(&filter);
    }
  }

  static const D3D_FEATURE_LEVEL feature_levels[] = {
      D3D_FEATURE_LEVEL_12_1,
      D3D_FEATURE_LEVEL_12_0,
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
  };

  D3D12_FEATURE_DATA_FEATURE_LEVELS feat_levels = {_countof(feature_levels), feature_levels, min_feature_level};

  hr = device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feat_levels, sizeof(feat_levels));

  D3D_FEATURE_LEVEL d3d_feature_level;

  if (SUCCEEDED(hr)) {
    d3d_feature_level = feat_levels.MaxSupportedFeatureLevel;
  } else {
    d3d_feature_level = min_feature_level;
  }

  pp_device.construct(adapters[device_index], d3d_feature_level, device);
  return S_OK;
}

HRESULT DXGIFactory::CreateSwapChain(const CommandQueue &command_queue,
                                     HWND hwnd,
                                     const DXGI_SWAP_CHAIN_DESC1 &desc,
                                     double_ptr<SwapChain> pp_swap_chain) {
  ComPtr<IDXGISwapChain1> swap_chain;
  RETURN_IF_FAILED_HR(
      factory_->CreateSwapChainForHwnd(command_queue.Handle(), hwnd, &desc, nullptr, nullptr, &swap_chain),
      "failed to create swap chain.");

  ComPtr<IDXGISwapChain3> swap_chain3;
  RETURN_IF_FAILED_HR(swap_chain.As(&swap_chain3), "failed to create swap chain 3.");

  if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
    // Enable HDR.
    DXGI_COLOR_SPACE_TYPE color_space = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
    UINT color_space_support = 0;
    if (SUCCEEDED(swap_chain3->CheckColorSpaceSupport(color_space, &color_space_support)) &&
        ((color_space_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) ==
         DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
      RETURN_IF_FAILED_HR(swap_chain3->SetColorSpace1(color_space), "failed to set color space.");
    }
  }

  pp_swap_chain.construct(swap_chain3);

  return S_OK;
}

HRESULT DXGIFactory::CreateSwapChain(const CommandQueue &command_queue,
                                     HWND hwnd,
                                     UINT buffer_count,
                                     DXGI_FORMAT format,
                                     double_ptr<SwapChain> pp_swap_chain) {
  // Get window size from hwnd.
  RECT rect;
  GetClientRect(hwnd, &rect);
  UINT width = rect.right - rect.left;
  UINT height = rect.bottom - rect.top;
  DXGI_SWAP_CHAIN_DESC1 desc = {};
  desc.BufferCount = buffer_count;
  desc.Width = width;
  desc.Height = height;
  desc.Format = format;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.SampleDesc.Count = 1;

#ifndef NDEBUG
  LogInfo("Swap Chain: {}x{}", width, height);
#endif

  return CreateSwapChain(command_queue, hwnd, desc, pp_swap_chain);
}

HRESULT DXGIFactory::CreateSwapChain(const CommandQueue &command_queue,
                                     HWND hwnd,
                                     UINT buffer_count,
                                     double_ptr<SwapChain> pp_swap_chain) {
  return CreateSwapChain(command_queue, hwnd, buffer_count, DXGI_FORMAT_R8G8B8A8_UNORM, pp_swap_chain);
}

HRESULT CreateDXGIFactory(DXGIFactoryCreateHint hint, double_ptr<DXGIFactory> pp_factory) {
  UINT dxgi_factory_flags = 0;

  // Enable the debug layer (requires the Graphics Tools "optional feature").
  // NOTE: Enabling the debug layer after device creation will invalidate the
  // active device.
  if (hint.enable_debug) {
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
      debugController->EnableDebugLayer();

      // Enable additional debug layers.
      dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
    }
  }

  ComPtr<IDXGIFactory4> factory;
  RETURN_IF_FAILED_HR(CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&factory)), "failed to create DXGI factory.");

  pp_factory.construct(hint, factory);
  return S_OK;
}
}  // namespace grassland::d3d12
