#include "grassland/rhi/backend/d3d12/d3d12_transient_descriptor_pool.h"

#include <algorithm>
#include <cassert>

namespace grassland::rhi::backend::d3d12 {
D3D12TransientDescriptorPool::Lease::Lease(D3D12TransientDescriptorPool *pool, ID3D12DescriptorHeap *heap, size_t level)
    : pool_(pool),
      heap_(heap),
      level_(level),
      count_(heap->GetDesc().NumDescriptors) {
}

D3D12TransientDescriptorPool::Lease::~Lease() {
  pool_->ReleasePage(heap_, level_);
}

ID3D12DescriptorHeap *D3D12TransientDescriptorPool::Lease::Heap() const {
  return heap_;
}

uint32_t D3D12TransientDescriptorPool::Lease::Count() const {
  return count_;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12TransientDescriptorPool::Lease::CPU(uint32_t index) const {
  assert(index < count_);
  auto result = heap_->GetCPUDescriptorHandleForHeapStart();
  result.ptr += SIZE_T(index) * pool_->stride_;
  return result;
}

D3D12_GPU_DESCRIPTOR_HANDLE D3D12TransientDescriptorPool::Lease::GPU(uint32_t index) const {
  assert(index < count_);
  auto result = heap_->GetGPUDescriptorHandleForHeapStart();
  result.ptr += UINT64(index) * pool_->stride_;
  return result;
}

D3D12TransientDescriptorPool::D3D12TransientDescriptorPool(ID3D12Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type)
    : device_(device),
      type_(type),
      limit_(type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? 2048 : 1000000),
      stride_(device->GetDescriptorHandleIncrementSize(type)) {
  assert(type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
}

void D3D12TransientDescriptorPool::Bucket::Record(size_t occupancy, std::chrono::steady_clock::time_point now) {
  // Allocate first so failure preserves previously recorded demand.
  peaks.emplace_back(occupancy, now);
  while (peaks.size() > 1 && peaks[peaks.size() - 2].first <= occupancy) {
    peaks.pop_back();
    peaks.back() = {occupancy, now};
  }
  while (now - peaks.front().second > std::chrono::seconds(1))
    peaks.pop_front();
}

ObjectPtr<D3D12TransientDescriptorPool::Lease> D3D12TransientDescriptorPool::Acquire(uint32_t minimum_count) {
  if (!minimum_count || minimum_count > limit_)
    return {};
  uint32_t capacity = 32;
  size_t level = 0;
  while (capacity < minimum_count) {
    capacity = std::min(capacity * 2, limit_);
    ++level;
  }
  ID3D12DescriptorHeap *heap = nullptr;
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &bucket = buckets_[level];
      if (bucket.available.empty()) {
        bucket.available.reserve(bucket.allocated + 1);
        ComPtr<ID3D12DescriptorHeap> storage;
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = type_;
        desc.NumDescriptors = capacity;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (!Check(device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&storage)), "Create transient descriptor page"))
          return {};
        auto pointer = storage.Get();
        bucket.pages.emplace(pointer, std::move(storage));
        bucket.available.push_back(pointer);
        ++bucket.allocated;
      }
      bucket.Record(bucket.occupied + 1, std::chrono::steady_clock::now());
      heap = bucket.available.back();
      bucket.available.pop_back();
      ++bucket.occupied;
    }
    return ObjectPtr<Lease>::Adopt(new Lease(this, heap, level));
  } catch (const std::bad_alloc &) {
    if (heap)
      ReleasePage(heap, level);
    return {};
  }
}

void D3D12TransientDescriptorPool::ReleasePage(ID3D12DescriptorHeap *heap, size_t level) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &bucket = buckets_[level];
  --bucket.occupied;
  try {
    bucket.Record(bucket.occupied, std::chrono::steady_clock::now());
  } catch (const std::bad_alloc &) {
    bucket.pages.erase(heap);
    --bucket.allocated;
    return;
  }
  bucket.available.push_back(heap);
  const auto demand = bucket.peaks.front().first;
  while (bucket.allocated > demand && !bucket.available.empty()) {
    bucket.pages.erase(bucket.available.back());
    bucket.available.pop_back();
    --bucket.allocated;
  }
}

std::pair<size_t, size_t> D3D12TransientDescriptorPool::GetPageCounts(uint32_t minimum_count) const {
  if (minimum_count > limit_)
    return {};
  std::lock_guard<std::mutex> lock(mutex_);
  if (!minimum_count) {
    std::pair<size_t, size_t> total{};
    for (const auto &bucket : buckets_) {
      total.first += bucket.allocated;
      total.second += bucket.occupied;
    }
    return total;
  }
  uint32_t capacity = 32;
  size_t level = 0;
  while (capacity < minimum_count) {
    capacity = std::min(capacity * 2, limit_);
    ++level;
  }
  return {buckets_[level].allocated, buckets_[level].occupied};
}

void D3D12TransientDescriptorPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &bucket : buckets_) {
    bucket.allocated -= bucket.available.size();
    for (auto heap : bucket.available)
      bucket.pages.erase(heap);
    bucket.available.clear();
    bucket.peaks.clear();
  }
}
}  // namespace grassland::rhi::backend::d3d12
