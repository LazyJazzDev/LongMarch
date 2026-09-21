#pragma once
#include "sparkium/core/code_lines.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::detail {
CodeLines CompileShaderGraph(const NodeValue &graph, const std::map<std::string, int> &textures);
}
