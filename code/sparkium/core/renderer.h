#pragma once

#include "sparkium/core/core.h"

namespace sparkium {

// Used on one rendering thread. Scene, Camera and Film remain owned by the
// caller; accepted commands retain their GPU resources through completion.
class Renderer {
 public:
  explicit Renderer(Core *core, rhi::QueueType queue = rhi::QUEUE_TYPE_ASSIST);

  bool IsValid() const;
  // Records the batch's immutable parameter snapshot; the returned group belongs
  // to this Renderer and is used by its path-tracing command recording.
  rhi::IBindingGroup *RecordParameters(rhi::ICommandContext *commands, const Scene *scene, const Film *film);
  rhi::ObjPtr<rhi::ISubmissionToken> Submit(Scene *scene,
                                            Camera *camera,
                                            Film *film,
                                            RenderPipeline pipeline = RENDER_PIPELINE_AUTO,
                                            rhi::IImage *output = nullptr,
                                            bool linear_hdr = false,
                                            const rhi::SubmitInfo &dependencies = {});

 private:
  Core *core_;
  rhi::ObjPtr<rhi::IQueue> queue_;
  rhi::ObjPtr<rhi::IBuffer> parameters_;
  rhi::ObjPtr<rhi::IBindingGroup> bindings_;
};

}  // namespace sparkium
