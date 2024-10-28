#include "grassland/d3d12/dxgi_factory.h"

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
  LogWarning("[Vulkan] " + message);
}

namespace {
std::string error_message;
}

void SetErrorMessage(const std::string &message) {
  LogError("[Vulkan] " + message);
  error_message = message;
}

std::string GetErrorMessage() {
  return error_message;
}

HRESULT CreateDXGIFactory(double_ptr<DXGIFactory> pp_factory) {
  UINT dxgi_factory_flags = 0;

#if defined(_DEBUG)
  // Enable the debug layer (requires the Graphics Tools "optional feature").
  // NOTE: Enabling the debug layer after device creation will invalidate the
  // active device.
  {
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
      debugController->EnableDebugLayer();

      // Enable additional debug layers.
      dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
    }
  }
#endif

  IDXGIFactory4 *factory;
  RETURN_IF_FAILED_HR(
      CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&factory)),
      "failed to create DXGI factory.");

  pp_factory.construct(factory);
  return S_OK;
}
}  // namespace grassland::d3d12
