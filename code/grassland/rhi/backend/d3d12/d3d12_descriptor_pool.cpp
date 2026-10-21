#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace grassland::rhi::backend::d3d12 {
D3D12DescriptorPool::Lease::Lease(D3D12DescriptorPool *pool, Page *page, uint32_t offset, uint32_t count)
    : pool_(pool),
      page_(page),
      offset_(offset),
      count_(count) {
}

D3D12DescriptorPool::Lease::~Lease() {
  pool_->ReleaseRange(page_, offset_, count_);
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

D3D12DescriptorPool::D3D12DescriptorPool(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t page_capacity)
    : device_(device),
      type_(type),
      page_capacity_(page_capacity),
      stride_(device->GetDescriptorHandleIncrementSize(type)) {
}

ObjectPtr<D3D12DescriptorPool::Lease> D3D12DescriptorPool::Acquire(uint32_t count) {
  if (!count || !page_capacity_)
    return {};
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    Page *page = nullptr;
    uint32_t offset = 0;
    uint32_t available = 0;
    // Geometric growth keeps pages ordered by capacity and their count logarithmic.
    for (auto &candidate : pages_) {
      auto fit = candidate->available.lower_bound({count, 0});
      if (fit != candidate->available.end()) {
        page = candidate.get();
        available = fit->first;
        offset = fit->second;
        break;
      }
    }
    if (!page) {
      uint64_t capacity = 1;
      const uint64_t minimum = pages_.empty() ? std::max(count, page_capacity_) : count;
      while (capacity < minimum)
        capacity *= 2;
      if (!pages_.empty())
        capacity = std::max(capacity, uint64_t(pages_.back()->capacity) * 2);
      if (capacity > std::numeric_limits<uint32_t>::max())
        return {};
      auto fresh = std::make_unique<Page>();
      fresh->capacity = uint32_t(capacity);
      D3D12_DESCRIPTOR_HEAP_DESC desc{};
      desc.Type = type_;
      desc.NumDescriptors = fresh->capacity;
      desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      if (!Check(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&fresh->heap)), "Create descriptor pool page"))
        return {};
      fresh->ranges.emplace(0, fresh->capacity);
      fresh->available.emplace(fresh->capacity, 0);
      page = fresh.get();
      available = fresh->capacity;
      pages_.push_back(std::move(fresh));
    }
    auto lease = ObjectPtr<Lease>::Adopt(new Lease(this, page, offset, count));
    auto range = page->ranges.extract(offset);
    auto fit = page->available.extract({available, offset});
    if (available > count) {
      range.key() = offset + count;
      range.mapped() = available - count;
      fit.value() = {available - count, offset + count};
      page->ranges.insert(std::move(range));
      page->available.insert(std::move(fit));
    }
    ++page->occupied;
    return lease;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void D3D12DescriptorPool::ReleaseRange(Page *page, uint32_t offset, uint32_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto right = page->ranges.lower_bound(offset);
  auto left = right == page->ranges.begin() ? page->ranges.end() : std::prev(right);
  if (left != page->ranges.end() && left->first + left->second == offset) {
    page->available.erase({left->second, left->first});
    offset = left->first;
    count += left->second;
    page->ranges.erase(left);
  }
  if (right != page->ranges.end() && offset + count == right->first) {
    page->available.erase({right->second, right->first});
    count += right->second;
    page->ranges.erase(right);
  }
  page->ranges.emplace(offset, count);
  page->available.emplace(count, offset);
  --page->occupied;
}

std::pair<size_t, size_t> D3D12DescriptorPool::GetPageCounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {pages_.size(),
          size_t(std::count_if(pages_.begin(), pages_.end(), [](const auto &page) { return page->occupied != 0; }))};
}
}  // namespace grassland::rhi::backend::d3d12
