#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_submission_token.h"

namespace grassland::rhi::backend {
struct Submission {
  ObjectPtr<CommandContext> commands;
  ObjectPtr<SubmissionToken> completion;
  TimelinePoint point{};
  std::vector<ObjectPtr<Event>> waits;
  bool failed{};
};
}  // namespace grassland::rhi::backend
