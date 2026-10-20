#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace grassland::rhi::backend::d3d12 {
D3D12DescriptorPool::Lease::Lease(D3D12DescriptorPool *pool,
                                  Page *page,
                                  uint32_t offset,
                                  uint32_t count,
                                  FreeRanges::node_type range_node,
                                  FreeIndex::node_type index_node)
    : pool_(pool),
      page_(page),
      offset_(offset),
      count_(count),
      range_node_(std::move(range_node)),
      index_node_(std::move(index_node)) {
}

D3D12DescriptorPool::Lease::~Lease() {
  pool_->ReleaseRange(page_, offset_, count_, std::move(range_node_), std::move(index_node_));
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
    auto selected = free_index_.end();
    // Skip insufficient ranges within each capacity class without scanning pages.
    for (auto first = free_index_.begin(); first != free_index_.end();) {
      const auto capacity = std::get<0>(first->first);
      auto fit = free_index_.lower_bound({capacity, count, 0, 0});
      if (fit != free_index_.end() && std::get<0>(fit->first) == capacity) {
        selected = fit;
        break;
      }
      first = fit;
    }
    if (selected == free_index_.end()) {
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
      fresh->id = pages_.size();
      D3D12_DESCRIPTOR_HEAP_DESC desc{};
      desc.Type = type_;
      desc.NumDescriptors = fresh->capacity;
      desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      if (!Check(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&fresh->heap)), "Create descriptor pool page"))
        return {};
      fresh->free_ranges.emplace(0, fresh->capacity);
      // Prepare both containers before registering the page in the pool.
      FreeIndex pending;
      pending.emplace(FreeKey{fresh->capacity, fresh->capacity, fresh->id, 0}, fresh.get());
      pages_.push_back(std::move(fresh));
      selected = free_index_.insert(pending.extract(pending.begin())).position;
    }
    Page *page = selected->second;
    const auto offset = std::get<3>(selected->first);
    const auto available = std::get<1>(selected->first);
    FreeRanges pending_range;
    FreeIndex pending_index;
    pending_range.emplace(offset, count);
    pending_index.emplace(FreeKey{page->capacity, count, page->id, offset}, page);
    // All potentially failing allocations precede the interval mutation.
    auto lease =
        ObjectPtr<Lease>::Adopt(new Lease(this, page, offset, count, pending_range.extract(pending_range.begin()),
                                          pending_index.extract(pending_index.begin())));
    auto range_node = page->free_ranges.extract(offset);
    auto index_node = free_index_.extract(selected);
    if (available > count) {
      range_node.key() = offset + count;
      range_node.mapped() = available - count;
      index_node.key() = {page->capacity, available - count, page->id, offset + count};
      page->free_ranges.insert(std::move(range_node));
      free_index_.insert(std::move(index_node));
    }
    ++page->occupied;
    return lease;
  } catch (const std::bad_alloc &) {
    return {};
  }
}

void D3D12DescriptorPool::ReleaseRange(Page *page,
                                       uint32_t offset,
                                       uint32_t count,
                                       FreeRanges::node_type range_node,
                                       FreeIndex::node_type index_node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto right = page->free_ranges.lower_bound(offset);
  if (right != page->free_ranges.begin()) {
    auto left = std::prev(right);
    if (left->first + left->second == offset) {
      free_index_.erase({page->capacity, left->second, page->id, left->first});
      offset = left->first;
      count += left->second;
      page->free_ranges.erase(left);
    }
  }
  if (right != page->free_ranges.end() && offset + count == right->first) {
    free_index_.erase({page->capacity, right->second, page->id, right->first});
    count += right->second;
    page->free_ranges.erase(right);
  }
  range_node.key() = offset;
  range_node.mapped() = count;
  index_node.key() = {page->capacity, count, page->id, offset};
  page->free_ranges.insert(std::move(range_node));
  free_index_.insert(std::move(index_node));
  --page->occupied;
}

std::pair<size_t, size_t> D3D12DescriptorPool::GetPageCounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {pages_.size(),
          size_t(std::count_if(pages_.begin(), pages_.end(), [](const auto &page) { return page->occupied != 0; }))};
}
}  // namespace grassland::rhi::backend::d3d12
