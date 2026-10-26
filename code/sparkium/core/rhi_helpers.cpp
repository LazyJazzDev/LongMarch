#include "sparkium/core/rhi_helpers.h"

namespace sparkium {
using namespace grassland::rhi;

bool Execute(IDevice *device, ICommandContext *commands) {
  if (!device || !commands)
    return false;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return false;
  ObjPtr<ISubmissionToken> submitted = queue->Submit(commands);
  if (!submitted)
    return false;
  return submitted->WaitHost();
}

bool UploadBuffer(IDevice *device, IBuffer *buffer, const void *data, uint64_t size, uint64_t offset) {
  if (!device || !buffer)
    return false;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  if (!commands)
    return false;
  if (!commands->CmdUploadBuffer(buffer, offset, data, size))
    return false;
  return Execute(device, commands.Get());
}

bool DownloadBuffer(IDevice *device, IBuffer *buffer, void *data, uint64_t size, uint64_t offset) {
  if (!device || !buffer)
    return false;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  if (!commands)
    return false;
  ObjPtr<IDownloadToken> download = commands->CmdDownloadBuffer(buffer, offset, size);
  if (!download)
    return false;
  if (!Execute(device, commands.Get()))
    return false;
  return download->CopyTo(data, size);
}

bool UploadImage(IDevice *device, IImage *image, const void *data) {
  if (!device || !image)
    return false;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  if (!commands)
    return false;
  const auto &desc = image->GetDesc();
  if (!commands->CmdUploadImage(image, data, uint64_t(desc.width) * ImageFormatSize(desc.format)))
    return false;
  return Execute(device, commands.Get());
}

bool DownloadImage(IDevice *device, IImage *image, void *data) {
  if (!device || !image)
    return false;
  ObjPtr<ICommandContext> commands = device->CreateCommandContext();
  if (!commands)
    return false;
  ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(image);
  if (!download)
    return false;
  if (!Execute(device, commands.Get()))
    return false;
  return download->CopyTo(data, download->GetSize());
}
}  // namespace sparkium
