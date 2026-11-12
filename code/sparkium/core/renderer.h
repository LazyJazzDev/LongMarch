#pragma once

#include <map>
#include <memory>

#include "sparkium/core/core.h"

namespace sparkium {

// Used on one rendering thread. Scene, Camera and Film remain owned by the
// caller; accepted commands retain their GPU resources through completion.
class Renderer {
 public:
  explicit Renderer(Core *core, rhi::QueueType queue = rhi::QUEUE_TYPE_ASSIST);

  bool IsValid() const;
  // Immutable descriptions identify a scene's shader/layout version. Programs
  // (including the native default SBT) belong to this Renderer; recorded commands
  // retain them independently. Descriptions borrow shaders from their scene.
  bool RecordProgram(rhi::ICommandContext *commands,
                     const std::shared_ptr<const rhi::RayTracingProgramDesc> &description);
  bool RecordProgram(rhi::ICommandContext *commands, const std::shared_ptr<const rhi::ComputeProgramDesc> &description);
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
  template <class Description, class Program>
  using ProgramCache = std::
      map<std::weak_ptr<const Description>, rhi::ObjPtr<Program>, std::owner_less<std::weak_ptr<const Description>>>;
  ProgramCache<rhi::RayTracingProgramDesc, rhi::IRayTracingProgram> ray_tracing_programs_;
  ProgramCache<rhi::ComputeProgramDesc, rhi::IComputeProgram> compute_programs_;
  Core *core_;
  rhi::ObjPtr<rhi::IQueue> queue_;
  rhi::ObjPtr<rhi::IBuffer> parameters_;
  rhi::ObjPtr<rhi::IBindingGroup> bindings_;
};

}  // namespace sparkium
