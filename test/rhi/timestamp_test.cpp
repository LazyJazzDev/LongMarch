#include <array>
#include <iostream>

#include "grassland/rhi/rhi.h"

using namespace grassland::rhi;

int main() {
  ObjPtr<ICore> core = CreateCore(BACKEND_API_D3D12, {});
  ObjPtr<IDevice> device = core ? core->CreateDeviceByAutoSelection() : nullptr;
  if (!device)
    return 1;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  const auto frequency = queue->TimestampFrequency();
  if (!frequency)
    return 2;
  ObjPtr<IImage> image = device->CreateImage({1024, 1024, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ObjPtr<IDownloadToken> begin = commands->CmdWriteTimestamp();
  if (!begin || begin->GetStatus() != SUBMISSION_STATUS_PENDING || begin->WaitHost())
    return 3;
  if (!commands->CmdClearImage(image.Get(), ClearValue{}))
    return 4;
  ObjPtr<IDownloadToken> end = commands->CmdWriteTimestamp();
  ObjPtr<ISubmissionToken> submitted = commands->Submit();
  SubmissionCpuTiming timing{};
  if (!end || !submitted || !commands->GetSubmissionCpuTiming(timing))
    return 5;
  commands.Release();  // Tokens retain query results after the recording handle dies.
  if (!submitted->WaitHost())
    return 6;
  uint64_t a = 0, b = 0;
  if (begin->GetSize() != 8 || end->GetSize() != 8 || !begin->CopyTo(&a, sizeof(a)) || !end->CopyTo(&b, sizeof(b)) ||
      b <= a)
    return 7;
  if (begin->CopyTo(nullptr, 8) || begin->CopyTo(&a, 4))
    return 8;
  std::cout << "GPU timestamp order, frequency, pending/completed readback and ownership passed: "
            << (b - a) * 1000. / frequency << " ms\n";
  return 0;
}
