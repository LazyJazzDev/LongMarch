#include "sparkium/pipelines/raster/core/geometry.h"

namespace sparkium::raster {

void Geometry::SetupProgram(rhi::ProgramDesc &program) {
}

bool Geometry::DispatchDrawCalls(rhi::ICommandContext *cmd_ctx) {
  return true;
}

}  // namespace sparkium::raster
