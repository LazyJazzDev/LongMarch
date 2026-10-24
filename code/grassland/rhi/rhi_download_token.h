#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
// A zero width/height extends to the corresponding edge of the image.
struct ImageDownloadRegion {
  uint32_t x{}, y{}, width{}, height{};
};

// Like other resource handles, tokens must be released before their Device.
// Submit before querying status or sharing the token with other threads.
// Recording cancellation is managed by the caller, not reported by this token.
class IDownloadToken : public IObject {
 public:
  // Without an associated SubmissionToken, the status is PENDING.
  virtual SubmissionStatus GetStatus() const = 0;
  // Returns false without a SubmissionToken; otherwise delegates to its WaitHost().
  virtual bool WaitHost() const = 0;
  virtual uint64_t GetSize() const = 0;
  // Caller must first observe COMPLETED (or a successful WaitHost) and perform
  // copies on one thread. No internal status check or serialization is performed.
  // Returns false for a null destination or insufficient destination_size.
  // Image rows are tightly packed in
  // the source format, top to bottom, with no row padding. D24S8 uses
  // a little-endian uint32: depth in bits 0..23, stencil in bits 24..31.
  // Metal converts its native D32S8 depth to the requested D24 precision.
  virtual bool CopyTo(void *destination, uint64_t destination_size) const = 0;

 protected:
  ~IDownloadToken() override;
};
}  // namespace grassland::rhi
