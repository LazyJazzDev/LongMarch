#pragma once
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {

class IImage : public IObject {
 public:
  explicit IImage(ImageDesc desc);
  const ImageDesc &GetDesc() const;
  virtual bool IsValid() const = 0;
  virtual const void *DeviceIdentity() const = 0;
  bool Claim();
  void CancelClaim();

 private:
  ImageDesc desc_;
  std::atomic<bool> claimed_{};
};

}  // namespace grassland::rhi::backend
