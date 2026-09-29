// Browser host: WebGPU on the page's canvas, the shared mobile demo sessions
// (DemoSession) and DOM input. The page selects the app with ?app=gol|2048|nbody.
#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "demos/DemoSession.h"
#include "grassland/graphics/backend/webgpu/webgpu_core.h"
#include "grassland/graphics/backend/webgpu/webgpu_image.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace {

constexpr const char *kCanvas = "#canvas";

// Copies the session image to the canvas. N-body renders linear light, which the
// SDR presentation encodes exactly like the HarmonyOS host.
constexpr const char *kBlitShader = R"(
@group(0) @binding(0) var source : texture_2d<f32>;

@vertex fn Vertex(@builtin(vertex_index) index : u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn Texel(position : vec4f) -> vec4f {
  let size = vec2i(textureDimensions(source)) - 1;
  return textureLoad(source, min(vec2i(position.xy), size), 0);
}

@fragment fn Copy(@builtin(position) position : vec4f) -> @location(0) vec4f {
  return vec4f(Texel(position).rgb, 1.0);
}

@fragment fn EncodeParticles(@builtin(position) position : vec4f) -> @location(0) vec4f {
  var rgb = pow(max(Texel(position).rgb, vec3f(0.0)), vec3f(2.2));
  rgb = clamp(rgb, vec3f(0.0), vec3f(1.0));
  rgb = select(1.055 * pow(rgb, vec3f(1.0 / 2.4)) - 0.055, rgb * 12.92, rgb <= vec3f(0.0031308));
  return vec4f(rgb, 1.0);
}
)";

struct Host {
  std::string app;
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Surface surface;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  wgpu::RenderPipeline copy, encode;
  wgpu::BindGroupLayout layout;
  std::unique_ptr<DemoSession> session;
  int width = 0, height = 0;
  double ratio = 1;
  // N-body camera, dragged like the mobile apps.
  float yaw = 0, pitch = 0;
  bool dragging = false;
  double drag_x = 0, drag_y = 0;
  // A Game of Life open/save request shown in the page's pattern library.
  bool file_pending = false;
  // Releases reach the game only after a press on the canvas, not from the page's sheet.
  bool pressed = false;
};

Host host;

void Fail(const std::string &message) {
  std::fprintf(stderr, "%s\n", message.c_str());
  EM_ASM(
      {
        if (Module.onLongMarchError)
          Module.onLongMarchError(UTF8ToString($0));
      },
      message.c_str());
  emscripten_cancel_main_loop();
}

bool Nbody() {
  return host.app == "nbody_cs";
}

grassland::graphics::Window *GameWindow() {
  auto game = host.session ? host.session->Game() : nullptr;
  return game ? game->Window() : nullptr;
}

void CreateBlit() {
  wgpu::ShaderSourceWGSL source{};
  source.code = kBlitShader;
  wgpu::ShaderModuleDescriptor module_descriptor{};
  module_descriptor.nextInChain = &source;
  auto module = host.device.CreateShaderModule(&module_descriptor);
  wgpu::BindGroupLayoutEntry entry{};
  entry.binding = 0;
  entry.visibility = wgpu::ShaderStage::Fragment;
  entry.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
  entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
  wgpu::BindGroupLayoutDescriptor layout_descriptor{};
  layout_descriptor.entryCount = 1;
  layout_descriptor.entries = &entry;
  host.layout = host.device.CreateBindGroupLayout(&layout_descriptor);
  wgpu::PipelineLayoutDescriptor pipeline_layout{};
  pipeline_layout.bindGroupLayoutCount = 1;
  pipeline_layout.bindGroupLayouts = &host.layout;
  auto layout = host.device.CreatePipelineLayout(&pipeline_layout);
  auto pipeline = [&](const char *fragment_entry) {
    wgpu::ColorTargetState target{};
    target.format = host.format;
    wgpu::FragmentState fragment{};
    fragment.module = module;
    fragment.entryPoint = fragment_entry;
    fragment.targetCount = 1;
    fragment.targets = &target;
    wgpu::RenderPipelineDescriptor descriptor{};
    descriptor.layout = layout;
    descriptor.vertex.module = module;
    descriptor.vertex.entryPoint = "Vertex";
    descriptor.fragment = &fragment;
    return host.device.CreateRenderPipeline(&descriptor);
  };
  host.copy = pipeline("Copy");
  host.encode = pipeline("EncodeParticles");
}

// Keeps the drawing buffer at device pixels; games receive the same size, so
// input coordinates are device pixels too.
void UpdateSize() {
  double css_width = 0, css_height = 0;
  emscripten_get_element_css_size(kCanvas, &css_width, &css_height);
  host.ratio = emscripten_get_device_pixel_ratio();
  const int width = std::max(1, int(std::lround(css_width * host.ratio)));
  const int height = std::max(1, int(std::lround(css_height * host.ratio)));
  if (width == host.width && height == host.height)
    return;
  host.width = width;
  host.height = height;
  emscripten_set_canvas_element_size(kCanvas, width, height);
  wgpu::SurfaceConfiguration configuration{};
  configuration.device = host.device;
  configuration.format = host.format;
  configuration.width = width;
  configuration.height = height;
  configuration.alphaMode = wgpu::CompositeAlphaMode::Opaque;
  host.surface.Configure(&configuration);
  host.session->Resize(width, height);
}

void Present(grassland::graphics::Image *image) {
  wgpu::SurfaceTexture surface_texture{};
  host.surface.GetCurrentTexture(&surface_texture);
  if (!surface_texture.texture)
    return;
  auto native = dynamic_cast<grassland::graphics::backend::WebGPUImage *>(image);
  if (!native)
    return;
  wgpu::BindGroupEntry entry{};
  entry.binding = 0;
  entry.textureView = native->View();
  wgpu::BindGroupDescriptor group_descriptor{};
  group_descriptor.layout = host.layout;
  group_descriptor.entryCount = 1;
  group_descriptor.entries = &entry;
  auto group = host.device.CreateBindGroup(&group_descriptor);
  wgpu::RenderPassColorAttachment attachment{};
  attachment.view = surface_texture.texture.CreateView();
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = {0, 0, 0, 1};
  wgpu::RenderPassDescriptor pass_descriptor{};
  pass_descriptor.colorAttachmentCount = 1;
  pass_descriptor.colorAttachments = &attachment;
  auto encoder = host.device.CreateCommandEncoder();
  auto pass = encoder.BeginRenderPass(&pass_descriptor);
  pass.SetPipeline(Nbody() ? host.encode : host.copy);
  pass.SetBindGroup(0, group);
  pass.Draw(3);
  pass.End();
  auto commands = encoder.Finish();
  host.device.GetQueue().Submit(1, &commands);
}

void Frame() {
  try {
    UpdateSize();
    if (Nbody())
      host.session->Configure(4096, 10, 0.03f, true, host.yaw, host.pitch, 0);
    if (auto game = host.session->Game()) {
      // The page shows its pattern library, then answers with lm_complete_file.
      const int request = game->FileRequest();
      if (request > 0 && !host.file_pending) {
        host.file_pending = true;
        EM_ASM({ Module.onLongMarchFile($0); }, request);
      }
    }
    host.session->Render();
    Present(host.session->Image());
  } catch (const std::exception &error) {
    Fail(error.what());
  }
}

// ---- Input: DOM events in CSS pixels become device pixels. ----

void Pointer(double x, double y) {
  if (auto window = GameWindow()) {
    host.session->Game()->PrepareInput(false);
    window->SendPointer(x * host.ratio, y * host.ratio);
  }
}

void Button(int button, bool press, double x, double y) {
  if (Nbody()) {
    host.dragging = press;
    host.drag_x = x;
    host.drag_y = y;
    return;
  }
  auto window = GameWindow();
  if (!window || (!press && !host.pressed))
    return;
  host.pressed = press;
  Pointer(x, y);
  if (press)
    window->CursorEnterEvent().InvokeCallbacks(true);
  window->SendMouseButton(button, press ? GLFW_PRESS : GLFW_RELEASE);
}

void Drag(double x, double y) {
  if (Nbody()) {
    if (host.dragging) {
      host.yaw += float(x - host.drag_x) * 0.01f;
      host.pitch = std::clamp(host.pitch + float(y - host.drag_y) * 0.01f, -1.5f, 1.5f);
      host.drag_x = x;
      host.drag_y = y;
    }
    return;
  }
  Pointer(x, y);
}

int MouseButton(const EmscriptenMouseEvent *event) {
  return event->button == 2 ? GLFW_MOUSE_BUTTON_RIGHT : GLFW_MOUSE_BUTTON_LEFT;
}

EM_BOOL OnMouseDown(int, const EmscriptenMouseEvent *event, void *) {
  Button(MouseButton(event), true, event->targetX, event->targetY);
  return EM_TRUE;
}

EM_BOOL OnMouseUp(int, const EmscriptenMouseEvent *event, void *) {
  Button(MouseButton(event), false, event->targetX, event->targetY);
  return EM_TRUE;
}

EM_BOOL OnMouseMove(int, const EmscriptenMouseEvent *event, void *) {
  Drag(event->targetX, event->targetY);
  return EM_TRUE;
}

EM_BOOL OnWheel(int, const EmscriptenWheelEvent *event, void *) {
  auto window = GameWindow();
  if (!window)
    return EM_TRUE;
  const double x = event->mouse.targetX * host.ratio, y = event->mouse.targetY * host.ratio;
  if (event->mouse.ctrlKey) {
    // Trackpad pinches arrive as ctrl + wheel.
    window->MagnifyEvent().InvokeCallbacks(grassland::graphics::MagnifyGesture{
        std::exp(-event->deltaY * 0.01), x, y, grassland::graphics::MagnifyPhase::kUpdate});
  } else {
    window->SendPointer(x, y);
    window->ScrollEvent().InvokeCallbacks(-event->deltaX / 100.0, -event->deltaY / 100.0);
  }
  return EM_TRUE;
}

// One finger acts as the left mouse button; the page disables browser gestures.
EM_BOOL OnTouch(int type, const EmscriptenTouchEvent *event, void *) {
  for (int i = 0; i < event->numTouches; ++i) {
    const auto &touch = event->touches[i];
    if (!touch.isChanged || touch.identifier != event->touches[0].identifier)
      continue;
    if (type == EMSCRIPTEN_EVENT_TOUCHSTART)
      Button(GLFW_MOUSE_BUTTON_LEFT, true, touch.targetX, touch.targetY);
    else if (type == EMSCRIPTEN_EVENT_TOUCHMOVE)
      Drag(touch.targetX, touch.targetY);
    else
      Button(GLFW_MOUSE_BUTTON_LEFT, false, touch.targetX, touch.targetY);
  }
  return EM_TRUE;
}

int Key(const EmscriptenKeyboardEvent *event) {
  const std::string key = event->code;
  if (key == "ArrowUp")
    return GLFW_KEY_UP;
  if (key == "ArrowDown")
    return GLFW_KEY_DOWN;
  if (key == "ArrowLeft")
    return GLFW_KEY_LEFT;
  if (key == "ArrowRight")
    return GLFW_KEY_RIGHT;
  if (key == "Space")
    return GLFW_KEY_SPACE;
  if (key == "Enter")
    return GLFW_KEY_ENTER;
  if (key == "Escape")
    return GLFW_KEY_ESCAPE;
  if (key.size() == 4 && key.rfind("Key", 0) == 0)
    return GLFW_KEY_A + (key[3] - 'A');
  return GLFW_KEY_UNKNOWN;
}

EM_BOOL OnKey(int type, const EmscriptenKeyboardEvent *event, void *) {
  // Typing in the page's pattern library stays in its text fields.
  if (EM_ASM_INT({
        const e = document.activeElement;
        return e && (e.tagName == 'INPUT' || e.tagName == 'TEXTAREA');
      }))
    return EM_FALSE;
  auto window = GameWindow();
  const int key = Key(event);
  if (!window || key == GLFW_KEY_UNKNOWN)
    return EM_FALSE;
  window->SendKey(key, type == EMSCRIPTEN_EVENT_KEYDOWN ? (event->repeat ? GLFW_REPEAT : GLFW_PRESS) : GLFW_RELEASE);
  return EM_TRUE;
}

void Start() {
  try {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas{};
    canvas.selector = kCanvas;
    wgpu::SurfaceDescriptor surface_descriptor{};
    surface_descriptor.nextInChain = &canvas;
    host.surface = host.instance.CreateSurface(&surface_descriptor);
    wgpu::SurfaceCapabilities capabilities{};
    host.surface.GetCapabilities(host.adapter, &capabilities);
    host.format = capabilities.formatCount ? capabilities.formats[0] : wgpu::TextureFormat::BGRA8Unorm;
    CreateBlit();
    grassland::graphics::backend::WebGPUCore::SetDevice(host.device);
    host.session = std::make_unique<DemoSession>("/res", host.app, false, grassland::graphics::BACKEND_API_WEBGPU);
    if (auto window = GameWindow()) {
      window->SendFocus(true);
      host.session->Game()->ResetClock();
    }
    emscripten_set_mousedown_callback(kCanvas, nullptr, EM_TRUE, OnMouseDown);
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, nullptr, EM_TRUE, OnMouseUp);
    emscripten_set_mousemove_callback(kCanvas, nullptr, EM_TRUE, OnMouseMove);
    emscripten_set_wheel_callback(kCanvas, nullptr, EM_TRUE, OnWheel);
    emscripten_set_touchstart_callback(kCanvas, nullptr, EM_TRUE, OnTouch);
    emscripten_set_touchmove_callback(kCanvas, nullptr, EM_TRUE, OnTouch);
    emscripten_set_touchend_callback(kCanvas, nullptr, EM_TRUE, OnTouch);
    emscripten_set_touchcancel_callback(kCanvas, nullptr, EM_TRUE, OnTouch);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, EM_TRUE, OnKey);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, EM_TRUE, OnKey);
    EM_ASM({
      if (Module.onLongMarchReady)
        Module.onLongMarchReady();
    });
    emscripten_set_main_loop(Frame, 0, false);
  } catch (const std::exception &error) {
    Fail(error.what());
  }
}

void RequestDevice() {
  wgpu::DeviceDescriptor descriptor{};
  descriptor.SetUncapturedErrorCallback([](const wgpu::Device &, wgpu::ErrorType, wgpu::StringView message) {
    std::fprintf(stderr, "WebGPU: %.*s\n", int(message.length), message.data);
  });
  host.adapter.RequestDevice(&descriptor, wgpu::CallbackMode::AllowSpontaneous,
                             [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                               if (status != wgpu::RequestDeviceStatus::Success) {
                                 Fail("WebGPU device unavailable: " + std::string(message.data, message.length));
                                 return;
                               }
                               host.device = std::move(device);
                               Start();
                             });
}

}  // namespace

// Completes the pending Game of Life file request: the game loads or saves the
// .cells file at path in the Emscripten file system, or cancels for "". Returns
// an error message, empty on success.
extern "C" EMSCRIPTEN_KEEPALIVE const char *lm_complete_file(const char *path) {
  static std::string error;
  host.file_pending = false;
  auto game = host.session ? host.session->Game() : nullptr;
  try {
    error = game ? game->CompleteFile(path) : "No game is running";
  } catch (const std::exception &e) {
    error = e.what();
  }
  return error.c_str();
}

int main(int argc, char **argv) {
  const std::string app = argc > 1 ? argv[1] : "gol";
  host.app = app == "nbody" ? "nbody_cs" : app;
  if (host.app != "gol" && host.app != "2048" && host.app != "nbody_cs") {
    Fail("Unknown app " + app);
    return 1;
  }
  host.instance = wgpu::CreateInstance(nullptr);
  host.instance.RequestAdapter(nullptr, wgpu::CallbackMode::AllowSpontaneous,
                               [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
                                 if (status != wgpu::RequestAdapterStatus::Success) {
                                   Fail("WebGPU is unavailable: " + std::string(message.data, message.length));
                                   return;
                                 }
                                 host.adapter = std::move(adapter);
                                 RequestDevice();
                               });
  return 0;
}
