#pragma once
#include <map>

#include "sparkium/pipelines/raytracing/core/core_util.h"

namespace sparkium::raytracing {

class Material : public Object {
 public:
  Material(Core *core);
  virtual ~Material() = default;

  virtual void Update(Scene *scene);
  virtual graphics::Buffer *Buffer() = 0;
  virtual const CodeLines &SamplerImpl() const = 0;
  virtual const CodeLines &EvaluatorImpl() const;

  // Graph parameters can be dispatched separately from the shared BSDF sampler.
  virtual const CodeLines *GraphImpl() const {
    return nullptr;
  }

 protected:
  // Material buffers contain CPU-owned data in fixed, non-overlapping ranges.
  // Compare the bytes last uploaded, so public-field edits and scene-specific
  // texture index changes still reach the GPU without explicit dirty flags.
  void UploadMaterialData(graphics::Buffer *buffer, const void *data, size_t size, size_t offset = 0);

  Core *core_;

 private:
  std::map<size_t, std::vector<uint8_t>> uploaded_data_;
};

}  // namespace sparkium::raytracing
