#pragma once
#include <array>
#include <memory>

#include "sparkium/core/core_util.h"

namespace sparkium {

class Film : public Object {
 public:
  Film(Core *core, int width, int height);

  // Two queued renders, the latest completed image and a UI-held image may coexist.
  static constexpr std::size_t kOutputSlots = 4;
  class Output;
  // Called on the rendering thread. Hold the lease through all rendering and
  // presentation uses, including GPU completion. Null with a valid Film means
  // all output slots are leased; resource creation failure invalidates Core.
  std::shared_ptr<const Output> AcquireOutput(bool linear_hdr = false);

  void Reset();
  bool IsValid() const;
  bool PrepareAccumulation();
  bool RecordReset(rhi::ICommandContext *commands);
  bool RecordResolve(rhi::ICommandContext *commands);
  bool RecordDevelop(rhi::ICommandContext *commands, rhi::IImage *target, bool linear_hdr = false);
  // Only the submission owner may advance the CPU accumulation state, after
  // the corresponding render submission has been accepted.
  void AcceptRender(int samples, rhi::IImage *output = nullptr);
  rhi::IBindingGroup *AccumulationGroup() const;

  class Output {
   public:
    rhi::IImage *Image() const {
      return image_.Get();
    }

   private:
    // Only Film allocates images and advances their accepted initialization state.
    friend std::shared_ptr<const Output> Film::AcquireOutput(bool);
    friend bool Film::RecordDevelop(rhi::ICommandContext *, rhi::IImage *, bool);
    friend void Film::AcceptRender(int, rhi::IImage *);
    Output() = default;
    rhi::ObjPtr<rhi::IImage> image_;
    bool initialized_{};
  };

  Core *GetCore() const;

  glm::uvec2 GetExtent();
  int GetWidth() const;
  int GetHeight() const;

  struct Info {
    int accumulated_samples{0};
    float persistence{1.0};
    float clamping{100.0f};
    float max_exposure{16.0f};
    int view_transform{0};
    float exposure{0.0f};
    float gamma{1.0f};
    float contrast{1.0f};
  } info;

  // HDR output is linear sRGB with artistic grading and an extended Filmic look.

  void RegisterResetCallback(const std::function<void()> &callback);

  rhi::IImage *GetRawImage() const;
  rhi::IImage *GetDepthImage() const;
  rhi::IImage *GetStencilImage() const;

 private:
  std::array<std::shared_ptr<Output>, kOutputSlots> outputs_;
  Core *core_;
  bool reset_pending_{true};
  rhi::ObjPtr<rhi::IImage> accumulated_color_, accumulated_samples_;
  rhi::ObjPtr<rhi::IBindingGroup> accumulation_group_, resolve_group_;

  glm::uvec2 extent_;
  rhi::ObjPtr<rhi::IImage> raw_image_;
  rhi::ObjPtr<rhi::IImage> depth_image_;
  rhi::ObjPtr<rhi::IImage> stencil_image_;
  rhi::ObjPtr<rhi::IBuffer> tone_mapping_buffer_;

  std::vector<std::function<void()>> reset_callbacks_;
  rhi::ObjPtr<rhi::IBindingGroup> tone_mapping_group_;
};

}  // namespace sparkium
