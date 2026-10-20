#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

#include <algorithm>
#include <cassert>

namespace grassland::rhi::backend::d3d12 {
D3D12DescriptorPool::Lease::Lease(D3D12DescriptorPool *pool,
                                  Page *page,
                                  Allocation *allocation,
                                  uint32_t offset,
                                  uint32_t count)
    : pool_(pool),
      page_(page),
      allocation_(allocation),
      offset_(offset),
      count_(count) {
}

D3D12DescriptorPool::Lease::~Lease() {
  pool_->ReleaseRange(page_, allocation_);
}

ID3D12DescriptorHeap *D3D12DescriptorPool::Lease::Heap() const {
  return page_->heap.Get();
}

uint32_t D3D12DescriptorPool::Lease::Count() const {
  return count_;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12DescriptorPool::Lease::CPU(uint32_t index) const {
  assert(index < count_);
  auto result = page_->heap->GetCPUDescriptorHandleForHeapStart();
  result.ptr += SIZE_T(offset_ + index) * pool_->stride_;
  return result;
}

D3D12_GPU_DESCRIPTOR_HANDLE D3D12DescriptorPool::Lease::GPU(uint32_t index) const {
  assert(index < count_ && pool_->shader_visible_);
  auto result = page_->heap->GetGPUDescriptorHandleForHeapStart();
  result.ptr += UINT64(offset_ + index) * pool_->stride_;
  return result;
}

D3D12DescriptorPool::D3D12DescriptorPool(ID3D12Device *device,
                                         D3D12_DESCRIPTOR_HEAP_TYPE type,
                                         bool shader_visible,
                                         uint32_t page_capacity)
    : device_(device),
      type_(type),
      shader_visible_(shader_visible),
      page_capacity_(page_capacity),
      stride_(device->GetDescriptorHandleIncrementSize(type)) {
}

ObjectPtr<D3D12DescriptorPool::Lease> D3D12DescriptorPool::Acquire(uint32_t count) {
  if (!count || !page_capacity_ ||
      (shader_visible_ && type_ != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
       type_ != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER))
    return {};
  const uint32_t limit = type_ == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? 2048 : 1000000;
  if (count > limit || page_capacity_ > limit)
    return {};
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    Page *page = nullptr;
    uint64_t start = 0;
    for (auto &candidate : pages_) {
      auto cursor = candidate->head;
      const auto remaining = candidate->capacity - cursor % candidate->capacity;
      if (count > remaining)
        cursor += remaining;
      if (cursor + count - candidate->tail <= candidate->capacity) {
        page = candidate.get();
        start = cursor;
        break;
      }
    }
    if (!page) {
      auto fresh = std::make_unique<Page>();
      fresh->capacity = std::max(count, page_capacity_);
      D3D12_DESCRIPTOR_HEAP_DESC desc{};
      desc.Type = type_;
      desc.NumDescriptors = fresh->capacity;
      desc.Flags = shader_visible_ ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      if (!Check(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&fresh->heap)), "Create descriptor pool page"))
        return {};
      page = fresh.get();
      pages_.push_back(std::move(fresh));
    }
    auto allocation = std::make_unique<Allocation>();
    allocation->end = start + count;
    auto pointer = allocation.get();
    page->allocations.push_back(std::move(allocation));
    // Commit the ring cursor only after the lease itself has been allocated.
    try {
      auto lease = ObjectPtr<Lease>::Adopt(new Lease(this, page, pointer, uint32_t(start % page->capacity), count));
      page->head = start + count;
      return lease;
    } catch (const std::bad_alloc &) {
      page->allocations.pop_back();
      return {};
    }
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void D3D12DescriptorPool::ReleaseRange(Page *page, Allocation *allocation) {
  std::lock_guard<std::mutex> lock(mutex_);
  allocation->released = true;
  while (!page->allocations.empty() && page->allocations.front()->released) {
    page->tail = page->allocations.front()->end;
    page->allocations.pop_front();
  }
  if (!page->allocations.empty())
    return;
  page->head = page->tail = 0;
  // Keep one ordinary idle page. Oversized pages are never retained as a cache.
  const bool spare = std::any_of(pages_.begin(), pages_.end(), [page, this](const auto &other) {
    return other.get() != page && other->capacity == page_capacity_ && other->allocations.empty();
  });
  if (page->capacity > page_capacity_ || spare)
    pages_.erase(std::find_if(pages_.begin(), pages_.end(), [page](const auto &entry) { return entry.get() == page; }));
}

void D3D12DescriptorPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  pages_.erase(std::remove_if(pages_.begin(), pages_.end(), [](const auto &page) { return page->allocations.empty(); }),
               pages_.end());
}

std::pair<size_t, size_t> D3D12DescriptorPool::GetPageCounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {pages_.size(), size_t(std::count_if(pages_.begin(), pages_.end(),
                                              [](const auto &page) { return !page->allocations.empty(); }))};
}
}  // namespace grassland::rhi::backend::d3d12
