#include "grassland/rhi/backend/backend_window_imgui.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>

#include "grassland/rhi/backend/backend_device.h"
#include "imgui_impl_glfw.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace grassland::rhi::backend {
namespace {
#ifdef _WIN32
constexpr wchar_t context_property[] = L"LongMarch.RHI.ImGuiContext";
constexpr wchar_t procedure_property[] = L"LongMarch.RHI.ImGuiWndProc";

LRESULT CALLBACK ImGuiWindowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
  auto context = reinterpret_cast<ImGuiContext *>(GetPropW(window, context_property));
  auto procedure = reinterpret_cast<WNDPROC>(GetPropW(window, procedure_property));
  ImGuiContextScope scope(context);
  return CallWindowProcW(procedure, window, message, wparam, lparam);
}
#endif

// RHI currently exposes float vertex attributes. Convert ImGui packed colors and
// positions while snapshotting its transient CPU lists for this recording.
struct Vertex {
  float position[2];
  float uv[2];
  float color[4];
};

constexpr const char *shader = R"(
RHI_TEXTURE2D(0, 0, float4, texture);
RHI_SAMPLER(0, 1, textureSampler);
struct Input {
  RHI_INPUT(0, float2, position);
  RHI_INPUT(1, float2, uv);
  RHI_INPUT(2, float4, color);
};
struct Varying {
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
  float4 color : COLOR0;
};
Varying VSMain(Input input) {
  Varying o;
  o.position = float4(input.position, 0, 1);
  o.uv = input.uv;
  o.color = input.color;
  return o;
}
float4 PSMain(Varying input) : SV_Target0 {
  return input.color * texture.Sample(textureSampler, input.uv);
}
)";
}  // namespace

std::unique_ptr<WindowImGui> WindowImGui::Create(Device *device,
                                                 GLFWwindow *window,
                                                 const char *font_file_path,
                                                 float font_size) {
  auto result = std::unique_ptr<WindowImGui>(new WindowImGui(device));
  if (!result->Initialize(window, font_file_path, font_size))
    return {};
  return result;
}

bool WindowImGui::Initialize(GLFWwindow *window, const char *font_file_path, float font_size) {
  if (!std::isfinite(font_size) || font_size <= 0)
    return false;
  // CreateContext selects the new context when none existed, so save before it.
  ImGuiContextScope scope(ImGui::GetCurrentContext());
  context_ = ImGui::CreateContext();
  ImGui::SetCurrentContext(context_);
  auto &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.BackendRendererName = "longmarch_rhi";
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  ImFontConfig font_config;
  font_config.SizePixels = font_size;
  if (font_file_path) {
    // Read explicitly: AddFontFromFileTTF asserts when a file cannot be opened.
    std::ifstream file(font_file_path, std::ios::binary | std::ios::ate);
    if (!file)
      return false;
    const auto size = file.tellg();
    if (size <= 100 || size > std::numeric_limits<int>::max())
      return false;
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(data.data(), static_cast<std::streamsize>(size)))
      return false;
    // ImGui copies non-owned data into its atlas, so the input can be released.
    font_config.FontDataOwnedByAtlas = false;
    if (!io.Fonts->AddFontFromMemoryTTF(data.data(), static_cast<int>(size), font_size, &font_config,
                                        io.Fonts->GetGlyphRangesChineseFull()))
      return false;
  } else {
    if (!io.Fonts->AddFontDefault(&font_config))
      return false;
  }
  if (!io.Fonts->Build())
    return false;
  platform_initialized_ = ImGui_ImplGlfw_InitForOther(window, false);
  if (!platform_initialized_)
    return false;
#ifdef _WIN32
  // The GLFW backend installs a Win32 hook even with install_callbacks=false.
  // Select this HWND's context before that hook reads BackendPlatformUserData;
  // native messages also arrive outside PollEvents and ImGui frame recording.
  auto native = static_cast<HWND>(ImGui::GetMainViewport()->PlatformHandleRaw);
  auto procedure = reinterpret_cast<WNDPROC>(GetWindowLongPtrW(native, GWLP_WNDPROC));
  if (!SetPropW(native, context_property, context_))
    return false;
  if (!SetPropW(native, procedure_property, reinterpret_cast<HANDLE>(procedure))) {
    RemovePropW(native, context_property);
    return false;
  }
  SetLastError(0);
  if (!SetWindowLongPtrW(native, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(ImGuiWindowProc)) && GetLastError()) {
    RemovePropW(native, procedure_property);
    RemovePropW(native, context_property);
    return false;
  }
#endif
  unsigned char *pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  font_ = device_->CreateImage({uint32_t(width), uint32_t(height), IMAGE_FORMAT_RGBA8_UNORM});
  sampler_ =
      device_->CreateSampler({SAMPLER_FILTER_LINEAR, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_CLAMP_TO_EDGE});
  if (!font_ || !sampler_)
    return false;
  ObjPtr<ICommandContext> upload = device_->CreateCommandContext();
  ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!upload || !queue || !upload->CmdUploadImage(font_.Get(), pixels, uint64_t(width) * 4))
    return false;
  ObjPtr<ISubmissionToken> token = queue->Submit(upload.Get());
  if (!token || !token->WaitHost())
    return false;
  io.Fonts->SetTexID(static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(font_.Get())));
  return true;
}

WindowImGui::~WindowImGui() {
  if (!context_)
    return;
  auto previous = ImGui::GetCurrentContext();
  ImGui::SetCurrentContext(context_);
  if (platform_initialized_) {
#ifdef _WIN32
    auto native = static_cast<HWND>(ImGui::GetMainViewport()->PlatformHandleRaw);
    if (auto procedure = RemovePropW(native, procedure_property))
      SetWindowLongPtrW(native, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(procedure));
    RemovePropW(native, context_property);
#endif
    ImGui_ImplGlfw_Shutdown();
  }
  ImGui::DestroyContext(context_);
  ImGui::SetCurrentContext(previous == context_ ? nullptr : previous);
}

bool WindowImGui::Record(ICommandContext *commands, IImage *image, EventManager<void()> &callbacks) {
  ImGuiContextScope scope(context_);
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  struct FrameScope {
    bool rendered{};

    ~FrameScope() {
      if (!rendered)
        ImGui::EndFrame();
    }
  } frame;

  callbacks.InvokeCallbacks();
  ImGui::Render();
  frame.rendered = true;
  return Render(commands, image, ImGui::GetDrawData());
}

bool WindowImGui::Render(ICommandContext *commands, IImage *image, ImDrawData *data) {
  if (!data || data->TotalVtxCount == 0 || data->TotalIdxCount == 0 || data->DisplaySize.x <= 0 ||
      data->DisplaySize.y <= 0)
    return true;
  const auto extent = image->GetDesc();
  if (!program_ || program_->GetDesc().color_formats[0] != extent.format) {
    ObjPtr<IShader> vs = device_->CreateShaderFromSource(shader, "VSMain", SHADER_STAGE_VERTEX);
    ObjPtr<IShader> ps = device_->CreateShaderFromSource(shader, "PSMain", SHADER_STAGE_FRAGMENT);
    if (!vs || !ps)
      return false;
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.vertex_bindings = {{sizeof(Vertex), false}};
    desc.vertex_attributes = {{0, 0, VERTEX_FORMAT_FLOAT2, offsetof(Vertex, position)},
                              {1, 0, VERTEX_FORMAT_FLOAT2, offsetof(Vertex, uv)},
                              {2, 0, VERTEX_FORMAT_FLOAT4, offsetof(Vertex, color)}};
    desc.color_formats = {extent.format};
    desc.blend_states = {BlendState(true)};
    desc.pipeline_layout.groups = {
        {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
    program_ = device_->CreateProgram(desc);
    if (!program_)
      return false;
  }
  std::vector<Vertex> vertices;
  std::vector<ImDrawIdx> indices;
  vertices.reserve(data->TotalVtxCount);
  indices.reserve(data->TotalIdxCount);
  for (const auto *list : data->CmdLists) {
    for (const auto &v : list->VtxBuffer) {
      const auto color = ImGui::ColorConvertU32ToFloat4(v.col);
      vertices.push_back({{(v.pos.x - data->DisplayPos.x) * 2 / data->DisplaySize.x - 1,
                           1 - (v.pos.y - data->DisplayPos.y) * 2 / data->DisplaySize.y},
                          {v.uv.x, v.uv.y},
                          {color.x, color.y, color.z, color.w}});
    }
    indices.insert(indices.end(), list->IdxBuffer.begin(), list->IdxBuffer.end());
  }
  // Each recording owns its allocations: later frames, abandoned recordings and
  // multiple in-flight submissions never overwrite this snapshot.
  ObjPtr<IBuffer> vb = device_->CreateBuffer({vertices.size() * sizeof(Vertex)});
  ObjPtr<IBuffer> ib = device_->CreateBuffer({indices.size() * sizeof(ImDrawIdx)});
  if (!vb || !ib || !commands->CmdUploadBuffer(vb.Get(), 0, vertices.data(), vertices.size() * sizeof(Vertex)) ||
      !commands->CmdUploadBuffer(ib.Get(), 0, indices.data(), indices.size() * sizeof(ImDrawIdx)))
    return false;
  auto setup = [&] {
    return commands->CmdBindProgram(program_.Get()) && commands->CmdBindVertexBuffer(0, vb.Get()) &&
           commands->CmdBindIndexBuffer(ib.Get(), sizeof(ImDrawIdx) == 2 ? INDEX_FORMAT_UINT16 : INDEX_FORMAT_UINT32) &&
           commands->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1});
  };
  if (!commands->CmdBindProgram(program_.Get()) ||
      !commands->CmdBeginRendering({{{image, ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_STORE_OP_STORE}}}))
    return false;
  bool success = setup();
  uint32_t index_offset = 0;
  int32_t vertex_offset = 0;
  for (const auto *list : data->CmdLists) {
    if (!success)
      break;
    for (const auto &draw : list->CmdBuffer) {
      if (draw.UserCallback) {
        if (draw.UserCallback != ImDrawCallback_ResetRenderState)
          draw.UserCallback(list, &draw);
        success = setup();
      } else {
        const auto clip = draw.ClipRect;
        const int x =
            int(std::clamp((clip.x - data->DisplayPos.x) * data->FramebufferScale.x, 0.0f, float(extent.width)));
        const int y =
            int(std::clamp((clip.y - data->DisplayPos.y) * data->FramebufferScale.y, 0.0f, float(extent.height)));
        const int right = int(
            std::clamp(std::ceil((clip.z - data->DisplayPos.x) * data->FramebufferScale.x), 0.0f, float(extent.width)));
        const int bottom = int(std::clamp(std::ceil((clip.w - data->DisplayPos.y) * data->FramebufferScale.y), 0.0f,
                                          float(extent.height)));
        if (right <= x || bottom <= y || !draw.ElemCount)
          continue;
        auto texture = reinterpret_cast<IImage *>(static_cast<uintptr_t>(draw.GetTexID()));
        ObjPtr<IBindingGroup> group = device_->CreateBindingGroup(program_->GetDesc().pipeline_layout.groups[0]);
        success =
            texture && group && group->BindResource(0, texture) && group->BindResource(1, sampler_.Get()) &&
            commands->CmdBindGroup(0, group.Get()) &&
            commands->CmdSetScissor({x, y, uint32_t(right - x), uint32_t(bottom - y)}) &&
            commands->CmdDrawIndexed(draw.ElemCount, 1, index_offset + draw.IdxOffset, vertex_offset + draw.VtxOffset);
      }
      if (!success)
        break;
    }
    index_offset += list->IdxBuffer.Size;
    vertex_offset += list->VtxBuffer.Size;
  }
  const bool ended = commands->CmdEndRendering();
  return success && ended;
}
}  // namespace grassland::rhi::backend
