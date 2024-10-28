#pragma once
#include <d3d12.h>
#include <d3dcompiler.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include "long_march.h"

namespace d3d12 {

inline std::string HrToString(HRESULT hr) {
  char s_str[64] = {};
  sprintf_s(s_str, "HRESULT of 0x%08X", static_cast<UINT>(hr));
  return std::string(s_str);
}

class HrException : public std::runtime_error {
 public:
  HrException(HRESULT hr) : std::runtime_error(HrToString(hr)), m_hr(hr) {
  }
  HRESULT Error() const {
    return m_hr;
  }

 private:
  const HRESULT m_hr;
};

#define SAFE_RELEASE(p) \
  if (p)                \
  (p)->Release()

inline void ThrowIfFailed(HRESULT hr) {
  if (FAILED(hr)) {
    throw HrException(hr);
  }
}

using Microsoft::WRL::ComPtr;
using namespace long_march;

class Application {
 public:
  Application();
  void Run();

 private:
  void OnInit();
  void OnUpdate();
  void OnRender();
  void OnClose();

  void CreateWindowAssets();
  void DestroyWindowAssets();

  GLFWwindow *glfw_window_;

  ComPtr<IDXGISwapChain3> swap_chain_;
};

}  // namespace d3d12
