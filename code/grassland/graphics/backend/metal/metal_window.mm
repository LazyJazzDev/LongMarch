#include "grassland/graphics/backend/metal/metal_window.h"

#include "grassland/graphics/backend/metal/metal_core.h"
#include "grassland/graphics/backend/metal/metal_image.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <QuartzCore/CAMetalLayer.h>

#include "backends/imgui_impl_glfw.h"
#define IMGUI_IMPL_METAL_CPP
#include "backends/imgui_impl_metal.h"

namespace grassland::graphics::backend {
MetalWindow::MetalWindow(MetalCore *core,
                         int width,
                         int height,
                         const std::string &title,
                         bool fullscreen,
                         bool resizable)
    : Window(width, height, title, fullscreen, resizable, false),
      core_(core) {
  MetalPool pool;
  auto view = [glfwGetCocoaWindow(GLFWWindow()) contentView];
  layer_ = NS::RetainPtr(CA::MetalLayer::layer());
  layer_->setDevice(core_->Device());
  layer_->setFramebufferOnly(true);
  [view setWantsLayer:YES];
  [view setLayer:(CAMetalLayer *)layer_.get()];
  ConfigurePresentation(false);
}

void MetalWindow::ConfigurePresentation(bool enable_hdr) {
  MetalPool pool;
  const auto format = enable_hdr ? MTL::PixelFormatRGBA16Float : MTL::PixelFormatBGRA8Unorm;
  const char *source = R"(
#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position [[position]]; float2 uv; };
vertex Vertex present_vertex(uint i [[vertex_id]]) {
  float2 p = float2((i << 1) & 2, i & 2);
  return {float4(p * float2(2, -2) + float2(-1, 1), 0, 1), p};
}
fragment float4 present_fragment(Vertex v [[stage_in]], texture2d<float> image [[texture(0)]]) {
  constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
  return image.sample(s, v.uv);
}
)";
  NS::Error *error = nullptr;
  auto library =
      NS::TransferPtr(core_->Device()->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), nullptr, &error));
  MetalCheck(library.get(), error, "presentation shaders");
  auto vertex = NS::TransferPtr(library->newFunction(NS::String::string("present_vertex", NS::UTF8StringEncoding)));
  auto fragment = NS::TransferPtr(library->newFunction(NS::String::string("present_fragment", NS::UTF8StringEncoding)));
  auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
  descriptor->setVertexFunction(vertex.get());
  descriptor->setFragmentFunction(fragment.get());
  descriptor->colorAttachments()->object(0)->setPixelFormat(format);
  auto pipeline = NS::TransferPtr(core_->Device()->newRenderPipelineState(descriptor.get(), &error));
  MetalCheck(pipeline.get(), error, "presentation pipeline");
  core_->WaitGPU();
  auto colorspace = CGColorSpaceCreateWithName(enable_hdr ? kCGColorSpaceExtendedLinearSRGB : kCGColorSpaceSRGB);
  if (!colorspace)
    throw std::runtime_error("Failed to create Metal presentation color space");
  auto native_layer = (CAMetalLayer *)layer_.get();
  native_layer.pixelFormat = (MTLPixelFormat)format;
  native_layer.colorspace = colorspace;
  native_layer.wantsExtendedDynamicRangeContent = enable_hdr;
  CGColorSpaceRelease(colorspace);
  pipeline_ = std::move(pipeline);
}

void MetalWindow::SetHDR(bool enable_hdr) {
  MetalPool pool;
  if (enable_hdr_ == enable_hdr)
    return;
  if (!GLFWWindow())
    throw std::runtime_error("Cannot change HDR on a closed Metal window");
  ConfigurePresentation(enable_hdr);
  Window::SetHDR(enable_hdr);
  if (enable_hdr) {
    auto screen = [glfwGetCocoaWindow(GLFWWindow()) screen];
    LogInfo("Metal HDR enabled: linear sRGB, RGBA16Float; display {} EDR headroom {:.2f}, potential {:.2f}",
            screen ? screen.localizedName.UTF8String : "unknown",
            double(screen.maximumExtendedDynamicRangeColorComponentValue),
            double(screen.maximumPotentialExtendedDynamicRangeColorComponentValue));
    if (screen.maximumPotentialExtendedDynamicRangeColorComponentValue <= 1.0)
      LogWarning("Current display has no HDR headroom; HDR output will be displayed within its SDR range");
  } else {
    LogInfo("Metal HDR disabled: sRGB, BGRA8Unorm");
  }
}

MetalWindow::~MetalWindow() {
  CloseWindow();
}

void MetalWindow::CloseWindow() {
  if (!GLFWWindow())
    return;
  MetalPool pool;
  core_->WaitGPU();
  // ImGui and the layer still need the native window during teardown.
  TerminateImGui();
  [[glfwGetCocoaWindow(GLFWWindow()) contentView] setLayer:nil];
  pipeline_.reset();
  layer_.reset();
  Window::CloseWindow();
}

void MetalWindow::InitImGui(const char *font, float size) {
  if (imgui_)
    return;
  MetalPool pool;
  imgui_ = ImGui::CreateContext();
  ImGui::SetCurrentContext(imgui_);
  if (font)
    ImGui::GetIO().Fonts->AddFontFromFileTTF(font, size);
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOther(GLFWWindow(), true);
  ImGui_ImplMetal_Init((__bridge id<MTLDevice>)core_->Device());
}

void MetalWindow::TerminateImGui() {
  if (!imgui_)
    return;
  core_->WaitGPU();
  ImGui::SetCurrentContext(imgui_);
  ImGui_ImplMetal_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext(imgui_);
  imgui_ = nullptr;
}

void MetalWindow::BeginImGuiFrame() {
  MetalPool pool;
  ImGui::SetCurrentContext(imgui_);
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  // ImGui only reads the attachment format/sample count here. The drawable arrives at Present.
  auto desc = MTL::TextureDescriptor::texture2DDescriptor(layer_->pixelFormat(), 1, 1, false);
  auto texture = NS::TransferPtr(core_->Device()->newTexture(desc));
  pass->colorAttachments()->object(0)->setTexture(texture.get());
  ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor *)pass);
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void MetalWindow::EndImGuiFrame() {
  ImGui::SetCurrentContext(imgui_);
  ImGui::Render();
}

void MetalWindow::Present(MTL::CommandBuffer *command, MetalImage *image) {
  MetalPool pool;
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return;
  layer_->setDrawableSize(CGSizeMake(width, height));
  auto drawable = layer_->nextDrawable();
  if (!drawable)
    return;
  auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
  auto attachment = pass->colorAttachments()->object(0);
  attachment->setTexture(drawable->texture());
  attachment->setLoadAction(MTL::LoadActionDontCare);
  attachment->setStoreAction(MTL::StoreActionStore);
  auto encoder = command->renderCommandEncoder(pass);
  encoder->setRenderPipelineState(pipeline_.get());
  encoder->setFragmentTexture(image->Handle(), 0);
  encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
  if (imgui_) {
    ImGui::SetCurrentContext(imgui_);
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), (__bridge id<MTLCommandBuffer>)command,
                                   (__bridge id<MTLRenderCommandEncoder>)encoder);
  }
  encoder->endEncoding();
  command->presentDrawable(drawable);
}
}  // namespace grassland::graphics::backend
