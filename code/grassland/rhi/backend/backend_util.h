#pragma once
#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

#include "grassland/rhi/backend/backend_query_interface.h"
#include "grassland/rhi/rhi_obj_ptr.h"

namespace grassland::rhi::backend {
class IBackendBindingGroup;
class IBackendBindingVersion;
class IBackendBuffer;
class IBackendCommandContext;
class IBackendComputeProgram;
class IBackendCore;
class IBackendDevice;
class IBackendDownloadToken;
class IBackendImage;
class IBackendProgram;
class IBackendQueue;
class IBackendRayTracingProgram;
class IBackendSampler;
class IBackendShader;
class IBackendShaderBindingTable;
class IBackendSubmissionToken;
class IBackendTimeline;
class IBackendWindow;
}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend::d3d12 {
class ID3D12AccelerationStructure;
class ID3D12AllocatedBuffer;
class ID3D12AllocatedImage;
class ID3D12BindingGroup;
class ID3D12BindingVersion;
class ID3D12BottomLevelAccelerationStructure;
class ID3D12Buffer;
class ID3D12CommandContext;
class ID3D12ComputeProgram;
class ID3D12Core;
class ID3D12Device;
class ID3D12DownloadToken;
class ID3D12Image;
class ID3D12Program;
class ID3D12ProgramBase;
class ID3D12Queue;
class ID3D12RayTracingProgram;
class ID3D12Sampler;
class ID3D12SwapChainImage;
class ID3D12TemporalBufferLease;
class ID3D12Timeline;
class ID3D12TopLevelAccelerationStructure;
class ID3D12Window;
}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::metal {
class IMetalAccelerationStructure;
class IMetalAllocatedBuffer;
class IMetalAllocatedImage;
class IMetalBindingGroup;
class IMetalBindingVersion;
class IMetalBottomLevelAccelerationStructure;
class IMetalBuffer;
class IMetalCommandContext;
class IMetalComputeProgram;
class IMetalCore;
class IMetalDevice;
class IMetalDownloadToken;
class IMetalImage;
class IMetalProgram;
class IMetalProgramBase;
class IMetalQueue;
class IMetalSampler;
class IMetalSwapChainImage;
class IMetalTemporalBufferLease;
class IMetalTimeline;
class IMetalTopLevelAccelerationStructure;
class IMetalWindow;
}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::vulkan {
class IVulkanAccelerationStructure;
class IVulkanAllocatedBuffer;
class IVulkanAllocatedImage;
class IVulkanBindingGroup;
class IVulkanBindingVersion;
class IVulkanBottomLevelAccelerationStructure;
class IVulkanBuffer;
class IVulkanCommandContext;
class IVulkanComputeProgram;
class IVulkanCore;
class IVulkanDevice;
class IVulkanDownloadToken;
class IVulkanImage;
class IVulkanProgram;
class IVulkanProgramBase;
class IVulkanQueue;
class IVulkanRayTracingProgram;
class IVulkanSampler;
class IVulkanSwapChainImage;
class IVulkanTemporalBufferLease;
class IVulkanTimeline;
class IVulkanTopLevelAccelerationStructure;
class IVulkanWindow;
}  // namespace grassland::rhi::backend::vulkan

namespace grassland::rhi::backend {
void ReportError(int32_t error_code, const std::string &error_message);

// Pointer adjustments stay in the query implementation, never at call sites.
template <typename Interface, typename Source>
void *MatchInterface(InterfaceID id, Source *source) noexcept {
  if constexpr (std::is_convertible_v<Source *, Interface *>) {
    if (id == uuid<Interface>())
      return static_cast<Interface *>(source);
  }
  return nullptr;
}

template <typename... Interfaces, typename Source>
void *MatchInterfaces(InterfaceID id, Source *source) noexcept {
  void *result = nullptr;
  ((result = MatchInterface<Interfaces>(id, source)) || ...);
  return result;
}

// One intrusive counter per native implementation, with no public state layout.
template <typename Interface>
class Object : public Interface {
 public:
  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  void AddRef() noexcept final {
    references_.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept final {
    if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }

  bool IsUniquelyOwned() const noexcept {
    return references_.load(std::memory_order_acquire) == 1;
  }

  ~Object() override = default;

 private:
  std::atomic<uint32_t> references_{1};
};

// Backend-only typed strong reference. T may be forward-declared at the storage site;
// define the owning class's constructor/destructor where T is complete.
template <typename T>
class ObjectPtr final {
 public:
  ObjectPtr() noexcept = default;

  // Consume the initial reference of a newly created object.
  static ObjectPtr Adopt(T *object) noexcept {
    static_assert(std::is_base_of_v<IObject, T>);
    ObjectPtr result;
    result.ptr_ = object;
    return result;
  }

  // The borrowed object must already be kept alive by another owner.
  static ObjectPtr Retain(T *object) noexcept {
    if (object)
      object->AddRef();
    return Adopt(object);
  }

  ObjectPtr(const ObjectPtr &other) noexcept : ObjectPtr(Retain(other.ptr_)) {
  }

  ObjectPtr(ObjectPtr &&other) noexcept : ptr_(other.Detach()) {
  }

  // Only implicit pointer conversions are accepted: derived -> base, never downcasts.
  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  ObjectPtr(const ObjectPtr<U> &other) noexcept : ObjectPtr(Retain(other.Get())) {
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  ObjectPtr(ObjectPtr<U> &&other) noexcept : ptr_(other.Detach()) {
  }

  ObjectPtr &operator=(ObjectPtr other) noexcept {
    Swap(other);
    return *this;
  }

  ~ObjectPtr() {
    if (ptr_)
      ptr_->Release();
  }

  T *Get() const noexcept {
    return ptr_;
  }

  T *operator->() const noexcept {
    return ptr_;
  }

  T &operator*() const noexcept {
    return *ptr_;
  }

  explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  void Reset() noexcept {
    ObjectPtr empty;
    Swap(empty);
  }

  void Swap(ObjectPtr &other) noexcept {
    std::swap(ptr_, other.ptr_);
  }

  // Transfer one owned reference without releasing it; the recipient must adopt it.
  T *Detach() noexcept {
    return std::exchange(ptr_, nullptr);
  }

 private:
  T *ptr_{nullptr};
};

class IBackendCore;
class IBackendDevice;

}  // namespace grassland::rhi::backend
