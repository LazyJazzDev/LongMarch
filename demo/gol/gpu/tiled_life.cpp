#include "gpu/tiled_life.h"

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
  batch_ = device->CreateBuffer({16});
  cells_ = device->CreateBuffer({uint64_t(WordCount()) * 4});
  decoded_ = device->CreateBuffer({uint64_t(cache_tiles_) * kTileSize * kTileSize});
  halos_ = device->CreateBuffer({uint64_t(tile_count_) * 33 * 4});
  PipelineLayoutDesc layout;
  layout.groups = {{{{0, BINDING_RW_BYTE_ADDRESS_BUFFER},
                     {1, BINDING_RW_BYTE_ADDRESS_BUFFER},
                     {2, BINDING_RW_BYTE_ADDRESS_BUFFER},
                     {3, BINDING_UNIFORM_BUFFER}}}};
  bindings_ = device->CreateBindingGroup(layout.groups[0]);
  if (!cells_ || !decoded_ || !halos_ || !batch_ || !bindings_ || !bindings_->BindResource(0, cells_.Get()) ||
      !bindings_->BindResource(1, decoded_.Get()) || !bindings_->BindResource(2, halos_.Get()) ||
      !bindings_->BindResource(3, batch_.Get()))
    return false;
  const char *entries[] = {"Decode", "Step", "ExportHalo"};
  for (uint32_t i = 0; i < programs_.size(); ++i) {
    ObjPtr<IShader> shader = device->CreateShaderFromSource(
        shader_source, entries[i], SHADER_STAGE_COMPUTE, {},
        {{"TILES_AXIS", std::to_string(tiles_per_axis)}, {"CACHE_TILES", std::to_string(cache_tiles_)}});
    if (!shader)
      return false;
    programs_[i] = device->CreateComputeProgram({shader.Get(), layout});
    if (!programs_[i])
      return false;
  }
  return true;
}

bool TiledLife::Dispatch(ICommandContext *commands, uint32_t phase, uint32_t count) {
  return commands && programs_[phase] && commands->CmdBindComputeProgram(programs_[phase].Get()) &&
         commands->CmdBindGroup(0, bindings_.Get()) && commands->CmdDispatch(count, 1, 1);
}

bool TiledLife::Upload(ICommandContext *commands, const std::vector<uint32_t> &packed) {
  return commands && cells_ && packed.size() == WordCount() &&
         commands->CmdUploadBuffer(cells_.Get(), 0, packed.data(), packed.size() * sizeof(uint32_t)) &&
         Dispatch(commands, 2, tile_count_);
}

bool TiledLife::Step(ICommandContext *commands) {
  // RHI inserts storage dependencies between dispatches. No CPU wait is needed.
  // In particular, no tile exports new borders until ALL main tiles are updated.
  for (uint32_t first = 0; first < tile_count_; first += cache_tiles_) {
    const std::array<uint32_t, 4> batch{first, 0, 0, 0};
    if (!commands || !commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) ||
        !Dispatch(commands, 0, cache_tiles_) || !Dispatch(commands, 1, cache_tiles_))
      return false;
  }
  return Dispatch(commands, 2, tile_count_);
}
}  // namespace life_demo
