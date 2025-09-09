#pragma once
#include "cao_di/d3d12/command_allocator.h"
#include "cao_di/d3d12/d3d12util.h"

namespace CD::d3d12 {

class AccelerationStructure {
 public:
  AccelerationStructure(Device *device, const ComPtr<ID3D12Resource> &as);

  ID3D12Resource *Handle() const {
    return as_.Get();
  }

  HRESULT UpdateInstances(const std::vector<D3D12_RAYTRACING_INSTANCE_DESC> &instances,
                          CommandQueue *queue,
                          Fence *fence,
                          CommandAllocator *allocator);

  HRESULT UpdateInstances(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                          CommandQueue *queue,
                          Fence *fence,
                          CommandAllocator *allocator);

 private:
  Device *device_;
  ComPtr<ID3D12Resource> as_;
};

}  // namespace CD::d3d12
