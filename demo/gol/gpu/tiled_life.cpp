#include "gpu/tiled_life.h"

#include "grassland/util/log.h"

namespace life_demo {
using namespace grassland::rhi;

bool TiledLife::Initialize(IDevice *device,
                           const std::string &shader_source,
                           uint32_t tiles_per_axis,
                           uint32_t compute_tile_size) {
  if (!device || !tiles_per_axis || tiles_per_axis > kTilesPerAxis || (tiles_per_axis & (tiles_per_axis - 1)))
    return false;
  if (compute_tile_size < 32 || compute_tile_size > 256 || (compute_tile_size & (compute_tile_size - 1)))
    return false;
  tile_count_ = tiles_per_axis * tiles_per_axis;
  compute_axis_ = tiles_per_axis * 256 / compute_tile_size;
  compute_count_ = compute_axis_ * compute_axis_;
  batch_ = device->CreateBuffer({256});
  cells_ = device->CreateBuffer({uint64_t(WordCount()) * 4});
  halos_ = device->CreateBuffer({uint64_t(compute_count_) * (4 * (compute_tile_size / 32) + 1) * 4});
  PipelineLayoutDesc layout;
  layout.groups = {{{{0, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                     {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  bindings_ = device->CreateBindingGroup(layout.groups[0]);
  if (!cells_ || !halos_ || !batch_ || !bindings_) {
    grassland::LogError("Allocate tiled GPU resources failed");
    return false;
  }
  IBuffer *buffers[] = {cells_.Get(), halos_.Get(), batch_.Get()};
  for (uint32_t i = 0; i < 3; ++i)
    if (!bindings_->BindResource(i, buffers[i])) {
      grassland::LogError("Bind tiled resource {} failed", i);
      return false;
    }
  const char *entries[] = {"Step", "ExportHalo", "Toggle", "Fill"};
  for (uint32_t i = 0; i < programs_.size(); ++i) {
    ObjPtr<IShader> shader = device->CreateShaderFromSource(
        shader_source, entries[i], SHADER_STAGE_COMPUTE, {},
        {{"TILES_AXIS", std::to_string(compute_axis_)},
         {"STORAGE_TILES_AXIS", std::to_string(tiles_per_axis)},
         {"TILE_SIZE", std::to_string(compute_tile_size)},
         {"GROUP_THREADS", std::to_string(std::min(256u, compute_tile_size * compute_tile_size / 32))}});
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
  if (!commands->CmdDispatch(count == compute_count_ ? compute_axis_ : count,
                             count == compute_count_ ? compute_axis_ : 1, 1)) {
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
  return Dispatch(commands, 1, compute_count_);
}

bool TiledLife::Step(ICommandContext *commands, bool periodic) {
  const std::array<uint32_t, 4> batch{0, 0, 0, periodic ? 1u : 0u};
  // The RHI storage dependency makes all main writes visible before exporting
  // the new borders. No CPU wait or global interior scratch buffer is needed.
  return commands && commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) &&
         Dispatch(commands, 0, compute_count_) && Dispatch(commands, 1, compute_count_);
}

bool TiledLife::Toggle(ICommandContext *commands, uint32_t x, uint32_t y) {
  const std::array<uint32_t, 4> batch{0, x, y, 0};
  return commands && commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) &&
         Dispatch(commands, 2, 1) && Dispatch(commands, 1, compute_count_);
}

bool TiledLife::Fill(ICommandContext *commands, bool random, uint32_t seed) {
  const std::array<uint32_t, 4> batch{0, seed, 0, random ? 1u : 0u};
  return commands && commands->CmdUploadBuffer(batch_.Get(), 0, batch.data(), sizeof(batch)) &&
         Dispatch(commands, 3, compute_count_) && Dispatch(commands, 1, compute_count_);
}
}  // namespace life_demo
