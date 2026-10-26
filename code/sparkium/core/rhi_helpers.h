#pragma once

#include "grassland/rhi/rhi.h"

namespace sparkium {
// Synchronous submission and CPU transfers for scene resources.
// Each helper waits for its submission before returning.
bool Execute(grassland::rhi::IDevice *device, grassland::rhi::ICommandContext *commands);
bool UploadBuffer(grassland::rhi::IDevice *device,
                  grassland::rhi::IBuffer *buffer,
                  const void *data,
                  uint64_t size,
                  uint64_t offset = 0);
bool DownloadBuffer(grassland::rhi::IDevice *device,
                    grassland::rhi::IBuffer *buffer,
                    void *data,
                    uint64_t size,
                    uint64_t offset = 0);
bool UploadImage(grassland::rhi::IDevice *device, grassland::rhi::IImage *image, const void *data);
bool DownloadImage(grassland::rhi::IDevice *device, grassland::rhi::IImage *image, void *data);
}  // namespace sparkium
