#pragma once

#include <array>

#include "grassland/rhi/rhi.h"

namespace life_demo {

// Bit-packed main tiles are updated in place. Each workgroup decodes its old
// tile into state and horizontal-count bit planes (or scans 2x2 via lookup).
// Each tile owns an input halo copied from neighboring main tiles after all
// updates complete; it stays unchanged throughout the next update dispatch.
// The default 64x64 compute tile uses 1.5 KiB of shared memory. Storage pages
// remain 256x256 independently of compute tiling. Unchanged tiles sleep once
// their eight neighbors are unchanged too; edits wake affected neighborhoods.
class TiledLife {
 public:
  enum class UpdateMethod { BitPlanes, Lookup2x2 };

  static constexpr uint32_t kTileSize = 256;
  static constexpr uint32_t kTilesPerAxis = 64;
  static constexpr uint32_t kWorldSize = kTileSize * kTilesPerAxis;
  static constexpr uint32_t kWordsPerTile = 2048;

  bool Initialize(grassland::rhi::IDevice *device,
                  const std::string &shader_source,
                  uint32_t tiles_per_axis = kTilesPerAxis,
                  uint32_t compute_tile_size = 64,
                  UpdateMethod method = UpdateMethod::BitPlanes,
                  bool sparse_updates = true,
                  bool allow_subgroup_vote = true);
  bool Upload(grassland::rhi::ICommandContext *commands, const std::vector<uint32_t> &packed);
  bool Step(grassland::rhi::ICommandContext *commands, bool periodic = true);
  bool Toggle(grassland::rhi::ICommandContext *commands, uint32_t x, uint32_t y);
  bool Fill(grassland::rhi::ICommandContext *commands, bool random, uint32_t seed = 0);

  // Rebuild only tiles changed since the last mip update, then reduce their
  // population totals. Call lazily for subpixel views; all work stays on GPU.
  bool UpdateMipmaps(grassland::rhi::ICommandContext *commands);

  grassland::rhi::IBuffer *Mipmaps() const {
    return mipmaps_.Get();
  }

  grassland::rhi::IBuffer *Cells() const {
    return cells_.Get();
  }

  uint32_t WordCount() const {
    return tile_count_ * kWordsPerTile;
  }

 private:
  bool Dispatch(grassland::rhi::ICommandContext *commands, uint32_t phase, uint32_t count);
  uint32_t tile_count_{};
  uint32_t compute_axis_{};
  uint32_t compute_count_{};
  uint32_t halo_groups_{};
  uint32_t tile_levels_{}, world_levels_{};
  bool mips_invalid_{true}, mips_current_{false}, sparse_updates_{true};
  grassland::rhi::ObjPtr<grassland::rhi::IBuffer> cells_, halos_, batch_, activity_, lookup_, mipmaps_;
  bool lookup_uploaded_{};
  bool last_periodic_{true};
  uint32_t generation_{};
  uint32_t NextGeneration();
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> bindings_;
  std::array<grassland::rhi::ObjPtr<grassland::rhi::IComputeProgram>, 6> programs_;
};

}  // namespace life_demo
