#pragma once

#include <array>

#include "grassland/rhi/rhi.h"

namespace life_demo {

// Bit-packed main tiles are updated in place. A byte-per-cell scratch buffer
// holds old interiors, and a separate packed buffer holds old edges/corners.
class TiledLife {
 public:
  static constexpr uint32_t kTileSize = 256;
  static constexpr uint32_t kTilesPerAxis = 64;
  static constexpr uint32_t kWorldSize = kTileSize * kTilesPerAxis;
  static constexpr uint32_t kWordsPerTile = 2048;

  bool Initialize(grassland::rhi::IDevice *device,
                  const std::string &shader_source,
                  uint32_t tiles_per_axis = kTilesPerAxis,
                  uint32_t cache_tiles = 256);
  bool Upload(grassland::rhi::ICommandContext *commands, const std::vector<uint32_t> &packed);
  bool Step(grassland::rhi::ICommandContext *commands, bool periodic = true);
  bool Toggle(grassland::rhi::ICommandContext *commands, uint32_t x, uint32_t y);
  bool Fill(grassland::rhi::ICommandContext *commands, bool random, uint32_t seed = 0);

  grassland::rhi::IBuffer *Cells() const {
    return cells_.Get();
  }

  uint32_t WordCount() const {
    return tile_count_ * kWordsPerTile;
  }

 private:
  bool Dispatch(grassland::rhi::ICommandContext *commands, uint32_t phase, uint32_t count);
  uint32_t tile_count_{};
  uint32_t cache_tiles_{};
  grassland::rhi::ObjPtr<grassland::rhi::IBuffer> cells_, decoded_, halos_, batch_;
  grassland::rhi::ObjPtr<grassland::rhi::IBindingGroup> bindings_;
  std::array<grassland::rhi::ObjPtr<grassland::rhi::IComputeProgram>, 5> programs_;
};

}  // namespace life_demo
