#include "imgui.h"

#include <iostream>
#include <vector>

#include "grassland/rhi/rhi_imgui.h"

using namespace grassland::rhi;

bool Frame(ImGuiRenderer &renderer, bool visible) {
  auto previous = ImGui::GetCurrentContext();
  ImGui::SetCurrentContext(renderer.Context());
  auto &io = ImGui::GetIO();
  io.DisplaySize = {128, 128};
  io.DisplayFramebufferScale = {1, 1};
  io.DeltaTime = 1.0f / 60.0f;
  const bool result = renderer.Update([&] {
    if (visible)
      ImGui::GetBackgroundDrawList()->AddRectFilled({16, 16}, {96, 96}, IM_COL32(255, 0, 0, 255));
  });
  const bool preserved = ImGui::GetCurrentContext() == renderer.Context();
  ImGui::SetCurrentContext(previous);
  return result && preserved;
}

bool Draw(IQueue *queue, IImage *image, ImGuiRenderer &renderer, bool red) {
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  if (!commands || !commands->CmdClearImage(image, {0, 0, 0, 1}) || !renderer.Record(commands.Get(), image))
    return false;
  ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(image);
  ObjPtr<ISubmissionToken> done = commands->Submit();
  if (!download || !done || !done->WaitHost())
    return false;
  std::vector<uint8_t> pixels(128 * 128 * 4);
  if (!download->CopyTo(pixels.data(), pixels.size()))
    return false;
  const auto inside = (32 * 128 + 32) * 4;
  const auto outside = (112 * 128 + 112) * 4;
  return pixels[inside] == (red ? 255 : 0) && pixels[inside + 1] == 0 && pixels[outside] == 0;
}

int main() {
  ObjPtr<ICore> core = CreateCore();
  ObjPtr<IDevice> device = core ? core->CreateDeviceByAutoSelection() : nullptr;
  if (!device)
    return 1;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IImage> image = device->CreateImage({128, 128, IMAGE_FORMAT_R8G8B8A8_UNORM});
  auto first = ImGuiRenderer::Create(device.Get());
  auto second = ImGuiRenderer::Create(device.Get());
  if (!queue || !image || !first || !second || first->Context() == second->Context())
    return 2;
  if (!Frame(*first, true) || !Frame(*second, false) || !Draw(queue.Get(), image.Get(), *first, true) ||
      !Draw(queue.Get(), image.Get(), *second, false))
    return 3;
  // Consumed draw data and an empty new frame must never replay old UI.
  if (!Draw(queue.Get(), image.Get(), *first, false) || !Frame(*first, false) ||
      !Draw(queue.Get(), image.Get(), *first, false))
    return 4;
  ImGui::SetCurrentContext(second->Context());
  first.reset();
  if (ImGui::GetCurrentContext() != second->Context())
    return 5;
  second.reset();
  if (ImGui::GetCurrentContext())
    return 6;
  std::cout << "PASS headless ImGui pixels, independent contexts, empty frames and destruction\n";
}
