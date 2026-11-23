#include "grassland/rhi/rhi_command_context.h"

#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi {
ICommandContext::~ICommandContext() = default;

bool ICommandContext::CmdUploadImage(IImage *image, const std::vector<ImageUploadInfo> &levels) {
  if (!image || levels.empty() || !image->IsValid())
    return false;
  for (const auto &upload : levels) {
    ImageDesc level;
    if (!ImageMipDesc(image->GetDesc(), upload.mip_level, level) || !upload.data || !IsColorFormat(level.format))
      return false;
    const uint64_t packed = uint64_t(level.width) * ImageFormatSize(level.format);
    if (!packed || upload.row_pitch < packed || (level.height - 1) > (SIZE_MAX - packed) / upload.row_pitch)
      return false;
  }
  for (const auto &upload : levels)
    if (!CmdUploadImage(image, upload.data, upload.row_pitch, upload.mip_level))
      return false;
  return true;
}
}  // namespace grassland::rhi
