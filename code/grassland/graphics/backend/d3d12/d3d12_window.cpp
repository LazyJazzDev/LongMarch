#include "grassland/graphics/backend/d3d12/d3d12_window.h"

#include "imgui_impl_dx12.h"
#include "imgui_impl_glfw.h"

namespace grassland::graphics::backend {
D3D12Window::D3D12Window(D3D12Core *core,
                         int width,
                         int height,
                         const std::string &title,
                         bool fullscreen,
                         bool resizable,
                         bool enable_hdr)
    : Window(width, height, title, fullscreen, resizable, enable_hdr), core_(core) {
  HWND hwnd = glfwGetWin32Window(GLFWWindow());
  core_->DXGIFactory()->CreateSwapChain(
      *core_->CommandQueue(), hwnd, std::max(std::min(core_->FramesInFlight(), DXGI_MAX_SWAP_CHAIN_BUFFERS), 2),
      enable_hdr_ ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM, &swap_chain_);
  ResizeEvent().RegisterCallback([this](int width, int height) {
    core_->WaitGPU();
    swap_chain_.reset();
    HWND hwnd = glfwGetWin32Window(GLFWWindow());
    core_->DXGIFactory()->CreateSwapChain(
        *core_->CommandQueue(), hwnd, std::max(std::min(core_->FramesInFlight(), DXGI_MAX_SWAP_CHAIN_BUFFERS), 2),
        enable_hdr_ ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM, &swap_chain_);
    if (imgui_assets_.context && imgui_assets_.rtv_format != swap_chain_->BackBufferFormat()) {
      ImGui_ImplDX12_Shutdown();
      ImGui_ImplGlfw_Shutdown();

      ImGui::DestroyContext(imgui_assets_.context);
      imgui_assets_.descriptor_alloc.Destroy();

      SetupImGuiContext();
    }
  });
}

D3D12Window::~D3D12Window() {
  if (GLFWWindow()) {
    D3D12Window::CloseWindow();
  }
}

void D3D12Window::CloseWindow() {
  core_->WaitGPU();
  if (imgui_assets_.context) {
    TerminateImGui();
  }
  swap_chain_.reset();
  Window::CloseWindow();
  ResizeEvent().UnregisterCallback(swap_chain_recreate_event_id_);
}

d3d12::SwapChain *D3D12Window::SwapChain() const {
  return swap_chain_.get();
}

ID3D12Resource *D3D12Window::CurrentBackBuffer() const {
  return swap_chain_->BackBuffer(swap_chain_->Handle()->GetCurrentBackBufferIndex());
}

void D3D12Window::InitImGui(const char *font_file_path, float font_size) {
  imgui_assets_.font_size = font_size;
  if (font_file_path) {
    imgui_assets_.font_path = font_file_path;
  }
  SetupImGuiContext();
}

void D3D12Window::TerminateImGui() {
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(imgui_assets_.context);
    imgui_assets_.context = nullptr;
  }
}

void D3D12Window::BeginImGuiFrame() {
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }
}

void D3D12Window::EndImGuiFrame() {
  if (imgui_assets_.context) {
    ImGui::SetCurrentContext(imgui_assets_.context);
    ImGui::Render();
    imgui_assets_.draw_command = true;
  }
}

ImGuiContext *D3D12Window::GetImGuiContext() const {
  return imgui_assets_.context;
}

D3D12ImGuiAssets &D3D12Window::ImGuiAssets() {
  return imgui_assets_;
}

void D3D12Window::SetupImGuiContext() {
  imgui_assets_.context = ImGui::CreateContext();
  ImGui::SetCurrentContext(imgui_assets_.context);
  ImGui::StyleColorsClassic();
  ImGui_ImplGlfw_InitForOther(GLFWWindow(), true);

  D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc = {};
  srv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  srv_heap_desc.NodeMask = 0;
  srv_heap_desc.NumDescriptors = 64;
  srv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  core_->Device()->CreateDescriptorHeap(srv_heap_desc, &imgui_assets_.srv_heap);
  imgui_assets_.descriptor_alloc.Create(core_->Device()->Handle(), imgui_assets_.srv_heap->Handle());

  ImGui_ImplDX12_InitInfo init_info = {};
  init_info.Device = core_->Device()->Handle();
  init_info.CommandQueue = core_->CommandQueue()->Handle();
  init_info.NumFramesInFlight = core_->FramesInFlight();
  init_info.RTVFormat = swap_chain_->BackBufferFormat();
  init_info.DSVFormat = DXGI_FORMAT_UNKNOWN;
  init_info.UserData = &imgui_assets_;

  init_info.SrvDescriptorHeap = imgui_assets_.srv_heap->Handle();
  init_info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo *info, D3D12_CPU_DESCRIPTOR_HANDLE *out_cpu_desc_handle,
                                      D3D12_GPU_DESCRIPTOR_HANDLE *out_gpu_desc_handle) {
    auto assets = static_cast<D3D12ImGuiAssets *>(info->UserData);
    assets->descriptor_alloc.Alloc(out_cpu_desc_handle, out_gpu_desc_handle);
  };
  init_info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo *info, D3D12_CPU_DESCRIPTOR_HANDLE cpu_desc_handle,
                                     D3D12_GPU_DESCRIPTOR_HANDLE gpu_desc_handle) {
    auto assets = static_cast<D3D12ImGuiAssets *>(info->UserData);
    assets->descriptor_alloc.Free(cpu_desc_handle, gpu_desc_handle);
  };
  ImGui_ImplDX12_Init(&init_info);

  imgui_assets_.rtv_format = swap_chain_->BackBufferFormat();

  auto &io = ImGui::GetIO();
  if (!imgui_assets_.font_path.empty()) {
    io.Fonts->AddFontFromFileTTF(imgui_assets_.font_path.c_str(), imgui_assets_.font_size, nullptr,
                                 io.Fonts->GetGlyphRangesChineseFull());
    io.Fonts->Build();
  } else {
    ImFontConfig im_font_config{};
    im_font_config.SizePixels = imgui_assets_.font_size;
    io.Fonts->AddFontDefault(&im_font_config);
  }

  ImGui_ImplDX12_CreateDeviceObjects();
  imgui_assets_.draw_command = false;
}

}  // namespace grassland::graphics::backend
