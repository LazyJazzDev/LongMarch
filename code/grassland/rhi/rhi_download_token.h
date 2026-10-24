#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
// A zero width/height extends to the corresponding edge of the image.
struct ImageDownloadRegion {
  uint32_t x{}, y{}, width{}, height{};
};

// Like other resource handles, tokens must be released before their Device.
class IDownloadToken : public IObject {
 public:
  virtual SubmissionStatus GetStatus() const = 0;
  virtual uint64_t GetSize() const = 0;
  // Nonblocking. Returns false without writing unless GPU execution succeeded
  // and destination_size covers GetSize(). Image rows are tightly packed in
  // the source format, top to bottom, with no row padding. D24S8 uses
  // a little-endian uint32: depth in bits 0..23, stencil in bits 24..31.
  // Metal converts its native D32S8 depth to the requested D24 precision.
  virtual bool CopyTo(void *destination, uint64_t destination_size) const = 0;

 protected:
  ~IDownloadToken() override = default;
};
}  // namespace grassland::rhi
