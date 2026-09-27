#pragma once

#include <map>

#include "sparkium/core/core_util.h"

namespace sparkium {

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
  // Native AS builders submit immediately; drain the batch only if their input is pending.
  void FlushBeforeRead(graphics::Buffer *buffer);

 private:
  friend class DataResource;
  void Register(DataResource *resource);
  void Unregister(DataResource *resource);
  DataResource &Find(const void *resource);
  graphics::Core *core_;
  std::map<const void *, DataResource *> resources_;
};

}  // namespace sparkium
