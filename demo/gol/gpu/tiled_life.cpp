#include "gpu/tiled_life.h"

#include "grassland/util/log.h"

namespace life_demo {
using namespace grassland::rhi;

bool TiledLife::Initialize(IDevice *device,
                           const std::string &shader_source,
                           uint32_t tiles_per_axis,
                           uint32_t cache_tiles) {
  if (!device || !tiles_per_axis || tiles_per_axis > kTilesPerAxis || (tiles_per_axis & (tiles_per_axis - 1)) ||
      !cache_tiles || cache_tiles > 256 || (cache_tiles & (cache_tiles - 1)))
    return false;
  tile_count_ = tiles_per_axis * tiles_per_axis;
  cache_tiles_ = std::min(tile_count_, cache_tiles);
  batch_ = device->CreateBuffer({256});
  cells_ = device->CreateBuffer({uint64_t(WordCount()) * 4});
  decoded_ = device->CreateBuffer({uint64_t(cache_tiles_) * kTileSize * kTileSize});
  halos_ = device->CreateBuffer({uint64_t(tile_count_) * 33 * 4});
  PipelineLayoutDesc layout;
  layout.groups = {{{{0, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                     {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                     {2, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                     {3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  bindings_ = device->CreateBindingGroup(layout.groups[0]);
  if (!cells_ || !decoded_ || !halos_ || !batch_ || !bindings_) {
    grassland::LogError("Allocate tiled resources failed: cells={} decoded={} halos={} batch={} bindings={}",
                        bool(cells_), bool(decoded_), bool(halos_), bool(batch_), bool(bindings_));
    return false;
  }
  IBuffer *buffers[] = {cells_.Get(), decoded_.Get(), halos_.Get(), batch_.Get()};
  for (uint32_t i = 0; i < 4; ++i)
    if (!bindings_->BindResource(i, buffers[i])) {
      grassland::LogError("Bind tiled resource {} size {} failed", i, buffers[i]->GetDesc().size);
      return false;
    }
  const char *entries[] = {"Decode", "Step", "ExportHalo", "Toggle", "Fill"};
  for (uint32_t i = 0; i < programs_.size(); ++i) {
    ObjPtr<IShader> shader = device->CreateShaderFromSource(
        shader_source, entries[i], SHADER_STAGE_COMPUTE, {},
        {{"TILES_AXIS", std::to_string(tiles_per_axis)}, {"CACHE_TILES", std::to_string(cache_tiles_)}});
    if (!shader) {
      grassland::LogError("Create tiled shader {} failed", entries[i]);
      return false;
    }
    programs_[i] = device->CreateComputeProgram({shader.Get(), layout});
    if (!programs_[i]) {
      grassland::LogError("Create tiled compute pipeline {} failed", entries[i]);
      return false;
    }
  }
  return true;
}

bool TiledLife::Dispatch(ICommandContext *commands, uint32_t phase, uint32_t count) {
  if (!commands || !programs_[phase])
    return false;
  if (!commands->CmdBindComputeProgram(programs_[phase].Get())) {
    grassland::LogError("Bind tiled pipeline {} failed", phase);
    return false;
  }
  if (!commands->CmdBindGroup(0, bindings_.Get())) {
    grassland::LogError("Bind tiled resources {} failed", phase);
    return false;
  }
  if (!commands->CmdDispatch(count, 1, 1)) {
    grassland::LogError("Dispatch tiled phase {} failed", phase);
    return false;
  }
  return true;
}

bool TiledLife::Upload(ICommandContext *commands, const std::vector<uint32_t> &packed) {
  if (!commands || !cells_ || packed.size() != WordCount()) {
    grassland::LogError("Invalid upload {} expected {}", packed.size(), WordCount());
    return false;
  }
  if (!commands->CmdUploadBuffer(cells_.Get(), 0, packed.data(), packed.size() * sizeof(uint32_t))) {
    grassland::LogError("Upload tiled grid failed");
    return false;
  }
  return Dispatch(commands, 2, tile_count_);
}

bool TiledLife::Step(ICommandContext *commands, bool periodic) {
  // RHI inserts storage dependencies between dispatches. No CPU wait is needed.
  // In particular, no tile exports new borders until ALL main tiles are updated.
  for (uint32_t first = 0; first < tile_count_; first += cache_tiles_) {
    const std::array<uint32_t, 4> batch{first, 0, 0, periodic ? 1u : 0u};
    if (!commands || !commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) ||
        !Dispatch(commands, 0, cache_tiles_) || !Dispatch(commands, 1, cache_tiles_))
      return false;
  }
  return Dispatch(commands, 2, tile_count_);
}

bool TiledLife::Toggle(ICommandContext *commands, uint32_t x, uint32_t y) {
  const std::array<uint32_t, 4> batch{0, x, y, 0};
  return commands && commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) &&
         Dispatch(commands, 3, 1) && Dispatch(commands, 2, tile_count_);
}

bool TiledLife::Fill(ICommandContext *commands, bool random, uint32_t seed) {
  const std::array<uint32_t, 4> batch{0, seed, 0, random ? 1u : 0u};
  return commands && commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) &&
         Dispatch(commands, 4, tile_count_) && Dispatch(commands, 2, tile_count_);
}
}  // namespace life_demo
