#include "sparkium/material/material_shader_graph.h"

namespace sparkium {

MaterialShaderGraph::MaterialShaderGraph(Core *core, const CodeLines &graph_code,
                                         const std::vector<graphics::Image *> &textures,
                                         const glm::vec3 &emission_hint)
    : Material(core), graph_code(graph_code), textures(textures), emission_hint(emission_hint) {}

}  // namespace sparkium
