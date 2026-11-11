#pragma once
#include "grassland/rhi/backend/backend_command_context.h"

namespace grassland::rhi::backend {
struct CommandContextState final {
  CommandContextState();
  ~CommandContextState();
  RenderingInfo rendering_{};
  bool recording_failed_{};
  bool sealed_{};
  Queue *queue_{};
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_{};
  bool submitted_{};
  std::vector<ObjectPtr<DownloadToken>> downloads_{};
};
}  // namespace grassland::rhi::backend
