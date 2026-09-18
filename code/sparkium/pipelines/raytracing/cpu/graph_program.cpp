#include "sparkium/pipelines/raytracing/cpu/graph_program.h"

#include "grassland/util/log.h"

namespace sparkium::raytracing::cpu {

void GraphProgramRegistry::Register(uint32_t material_data_index, std::unique_ptr<GraphProgram> program) {
  if (programs_.size() <= material_data_index) {
    programs_.resize(material_data_index + 1);
    reported_.resize(material_data_index + 1, false);
  }
  programs_[material_data_index] = std::move(program);
}

void GraphProgramRegistry::Clear() {
  programs_.clear();
  reported_.clear();
}

bool GraphProgramRegistry::HasProgram(uint32_t material_data_index) const {
  return material_data_index < programs_.size() && programs_[material_data_index] != nullptr;
}

void GraphProgramRegistry::Evaluate(uint32_t material_data_index,
                                    const GraphEvalInput &input,
                                    GraphEvalOutput &output) const {
  if (!HasProgram(material_data_index)) {
    // Report once per material rather than once per hit; a shaded surface with
    // no program means the scene was not prepared, which is worth seeing.
    if (reported_.size() <= material_data_index)
      reported_.resize(material_data_index + 1, false);
    if (!reported_[material_data_index]) {
      reported_[material_data_index] = true;
      grassland::LogError("[sparkium] no CPU shader-graph program registered for material buffer {}",
                          material_data_index);
    }
    return;
  }
  programs_[material_data_index]->Evaluate(input, output);
}

}  // namespace sparkium::raytracing::cpu
