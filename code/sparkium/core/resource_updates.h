#pragma once

#include <functional>
#include <unordered_set>
#include <variant>

#include "grassland/rhi/rhi_buffer.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_image.h"
#include "grassland/rhi/rhi_obj_ptr.h"

namespace sparkium {

// Scene construction and edits enqueue immutable CPU snapshots. The renderer
// records them into its own context and acknowledges them only after Submit
// accepts that context. This object never creates a context, submits or waits.
// All operations belong to the renderer thread.
class ResourceUpdates {
 public:
  bool Upload(grassland::rhi::IBuffer *buffer, const void *data, uint64_t size, uint64_t offset = 0);
  bool Upload(grassland::rhi::IImage *image, const void *data, uint64_t row_pitch = 0);
  bool Build(grassland::rhi::IBottomLevelAccelerationStructure *structure,
             const std::vector<grassland::rhi::AccelerationStructureGeometryBuffers> &geometries);

  class Cursor;
  Cursor Begin() const;
  // May be called again with the same cursor/context after scene preparation
  // enqueues more uploads. Failed recording must be discarded; Begin a fresh
  // cursor to retry all unacknowledged data in another context.
  bool Record(grassland::rhi::ICommandContext *commands, Cursor &cursor) const;
  bool Accept(const Cursor &cursor);
  bool Empty() const;

  class Cursor {
   public:
    // Cache state becomes reusable only when this recording is accepted. Capture
    // owned state, not a borrowed scene object: a canceled recording may outlive it.
    void OnAccepted(std::function<void()> action);
    bool Builds(grassland::rhi::IBottomLevelAccelerationStructure *structure) const;
    bool Writes(grassland::rhi::IBuffer *buffer) const;

   private:
    // Only the owner initializes and advances recording/acceptance bookkeeping.
    friend Cursor ResourceUpdates::Begin() const;
    friend bool ResourceUpdates::Record(grassland::rhi::ICommandContext *, Cursor &) const;
    friend bool ResourceUpdates::Accept(const Cursor &);
    uint64_t first{}, revision{};
    size_t count{};
    std::unordered_set<grassland::rhi::IBottomLevelAccelerationStructure *> builds;
    std::unordered_set<grassland::rhi::IBuffer *> writes;
    std::vector<std::function<void()>> accepted;
  };

 private:
  struct BufferUpload {
    grassland::rhi::ObjPtr<grassland::rhi::IBuffer> buffer;
    uint64_t offset;
    std::vector<uint8_t> bytes;
  };

  struct ImageUpload {
    grassland::rhi::ObjPtr<grassland::rhi::IImage> image;
    uint64_t row_pitch;
    std::vector<uint8_t> bytes;
  };

  struct GeometryBuild {
    grassland::rhi::ObjPtr<grassland::rhi::IBottomLevelAccelerationStructure> structure;
    std::vector<grassland::rhi::AccelerationStructureGeometryBuffers> geometries;
    std::vector<grassland::rhi::ObjPtr<grassland::rhi::IBuffer>> buffers;
  };

  std::vector<std::variant<BufferUpload, ImageUpload, GeometryBuild>> uploads_;
  uint64_t first_{};
  uint64_t revision_{};
};

}  // namespace sparkium
