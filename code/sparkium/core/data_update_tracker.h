#pragma once

#include <map>
#include <set>

#include "grassland/grassland.h"

namespace sparkium {
namespace graphics = grassland::graphics;
class Buffer;
class Image;
class BottomLevelAccelerationStructure;
class TopLevelAccelerationStructure;

// Used on the render thread, like the graphics Core.
// CPU-owned ranges only: GPU-written portions of a resource are never mirrored
// or overwritten. Resources must outlive submission, as with other GPU commands.
class DataUpdateTracker {
 public:
  explicit DataUpdateTracker(graphics::Core *core) : core_(core) {
  }

  ~DataUpdateTracker();
  DataUpdateTracker(const DataUpdateTracker &) = delete;
  DataUpdateTracker &operator=(const DataUpdateTracker &) = delete;
  // External graphics targets have no registration to invalidate.
  void InvalidateIfTracked(graphics::Image *image);
  void Update(graphics::Buffer *buffer, const void *data, size_t size, size_t offset = 0);
  void Update(graphics::Image *image, const void *data);
  void Update(graphics::Image *image, const void *data, graphics::Offset2D offset, graphics::Extent2D extent);
  // Explicit invalidation is required after resizing or writing tracked ranges
  // outside this tracker. GPU writes to disjoint ranges need no invalidation.
  void Invalidate(graphics::Buffer *buffer);
  void Invalidate(graphics::Image *image);
  void Flush();

  uint64_t Revision(graphics::Buffer *buffer);
  bool Contains(BottomLevelAccelerationStructure *blas) const;

  graphics::Core *GetCore() const {
    return core_;
  }

  // Registration accepts only resources created for this tracker. Unregistering
  // detaches the resource and invalidates dependents; detached resources cannot rejoin.
  void Register(BottomLevelAccelerationStructure *blas);
  void Register(TopLevelAccelerationStructure *tlas);
  void Unregister(BottomLevelAccelerationStructure *blas);
  void Unregister(TopLevelAccelerationStructure *tlas);
  void Register(Buffer *buffer);
  void Register(Image *image);
  void Unregister(Buffer *buffer);
  void Unregister(Image *image);

 private:
  Buffer &Find(graphics::Buffer *buffer);
  Image &Find(graphics::Image *image);
  graphics::Core *core_;
  std::map<graphics::Buffer *, Buffer *> buffers_;
  std::map<graphics::Image *, Image *> images_;
  std::set<BottomLevelAccelerationStructure *> blases_;
  std::set<TopLevelAccelerationStructure *> tlases_;
};

}  // namespace sparkium
