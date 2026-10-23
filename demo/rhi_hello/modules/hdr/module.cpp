#include "../../module.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"
}

int RunHDR(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;

  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  ObjPtr<IWindow> window;
  std::vector<ColorSpace> spaces{COLOR_SPACE_SRGB_NONLINEAR};
  if (!headless) {
    window = device->CreateGLFWWindow({960, 540, "RHI HDR - H: color space, Escape: close"});
    if (!window)
      return 1;
    for (const auto &option : window->GetSupportedPresentationOptions()) {
      std::cout << ColorSpaceName(option.color_space) << " / " << ImageFormatName(option.format) << " / "
                << PresentModeName(option.present_mode) << '\n';
      if (std::find(spaces.begin(), spaces.end(), option.color_space) == spaces.end())
        spaces.push_back(option.color_space);
    }
    if (spaces.size() == 1)
      std::cout << "HDR presentation unavailable on this surface; showing SDR explicitly.\n";
  } else {
    spaces = {COLOR_SPACE_SRGB_NONLINEAR, COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_HDR10_ST2084};
  }
  std::cout
      << "hdr: nominal columns 20, 80, 203, 400, 1000 nits; PQ uses 100 nits/unit (1.25x): 25, 100, 253.75, 500, 1250 nits; top neutral, bottom Rec.709 primaries.\n";
  const auto pq = std::find(spaces.begin(), spaces.end(), COLOR_SPACE_HDR10_ST2084);
  size_t selected = pq != spaces.end() ? static_cast<size_t>(pq - spaces.begin()) : (spaces.size() > 1 ? 1 : 0);
  if (window && !window->SetColorSpace(spaces[selected]))
    return 1;
  const auto &embedded = shader_list.at("modules/hdr/shaders/hdr.slang");
  const std::string shader(embedded.first, embedded.second);
  std::map<std::pair<ImageFormat, ColorSpace>, ObjPtr<IProgram>> programs;
  std::map<ColorSpace, ObjPtr<IImage>> targets;
  bool previous_h = false;
  int rendered = 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!PollEvents() || window->ShouldClose() || window->IsKeyDown(GLFW_KEY_ESCAPE))
        break;
      if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
        return 1;
      const bool h = window->IsKeyDown(GLFW_KEY_H);
      // Bounded runs exercise all available color spaces automatically.
      const bool cycle = frames >= 9 && rendered > 0 && rendered % std::max(1, frames / 3) == 0;
      if ((h && !previous_h) || cycle) {
        selected = (selected + 1) % spaces.size();
        if (!window->SetColorSpace(spaces[selected]))
          return 1;
      }
      previous_h = h;
    } else {
      selected = (rendered / std::max(1, frames / 3)) % spaces.size();
    }
    const auto color_space = headless ? spaces[selected] : window->GetColorSpace();
    ObjPtr<IImage> image;
    if (headless) {
      auto &target = targets[color_space];
      if (!target) {
        const auto format = color_space == COLOR_SPACE_EXTENDED_SRGB_LINEAR ? IMAGE_FORMAT_RGBA16_FLOAT
                            : color_space == COLOR_SPACE_HDR10_ST2084       ? IMAGE_FORMAT_RGB10A2_UNORM
                                                                            : IMAGE_FORMAT_RGBA8_UNORM;
        target = device->CreateImage({640, 360, format});
        if (!target)
          return 1;
      }
      image = target;
    } else
      image = window->GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    const auto extent = image->GetDesc();
    auto &program = programs[{extent.format, color_space}];
    if (!program) {
      const auto source = "#define OUTPUT_COLOR_SPACE " + std::to_string(static_cast<int>(color_space)) + "\n" + shader;
      ObjPtr<IShader> vs = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
      ObjPtr<IShader> ps = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
      if (!vs || !ps)
        return 1;
      ProgramDesc desc;
      desc.vertex_shader = vs.Get();
      desc.fragment_shader = ps.Get();
      desc.color_formats = {extent.format};
      program = device->CreateProgram(desc);
      if (!program)
        return 1;
      std::cout << "Rendering " << ColorSpaceName(color_space) << " / " << ImageFormatName(extent.format) << '\n';
    }
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context || !context->CmdBindProgram(program.Get()) ||
        !context->CmdBeginRendering(
            {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}) ||
        !context->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
        !context->CmdSetScissor({0, 0, extent.width, extent.height}) || !context->CmdDraw(3) ||
        !context->CmdEndRendering())
      return 1;
    if (!headless && !context->CmdPresent(window.Get()))
      return 1;
    ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());
    if (!token || !token->WaitHost())
      return 1;
    // The image and recording leave scope before a color-space switch or PollEvents.
    if (!headless && !window->SetTitle("RHI HDR - " + ColorSpaceName(color_space) + " - H: color space"))
      return 1;
    ++rendered;
  }
  std::cout << "hdr: completed " << rendered << " frames ("
            << (headless ? "offscreen encoding; no HDR display validation" : "window presentation") << ").\n";
  return 0;
}
