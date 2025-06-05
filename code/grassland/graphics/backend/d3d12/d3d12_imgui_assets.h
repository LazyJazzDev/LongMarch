#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"
#include "imgui_impl_dx12.h"

namespace grassland::graphics::backend {

struct ExampleDescriptorHeapAllocator {
  ID3D12DescriptorHeap *heap = nullptr;
  D3D12_DESCRIPTOR_HEAP_TYPE heap_type = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES;
  D3D12_CPU_DESCRIPTOR_HANDLE heap_start_cpu;
  D3D12_GPU_DESCRIPTOR_HANDLE heap_start_gpu;
  UINT heap_handle_increment;
  ImVector<int> free_indices;

  void Create(ID3D12Device *device, ID3D12DescriptorHeap *heap) {
    IM_ASSERT(this->heap == nullptr && free_indices.empty());
    this->heap = heap;
    D3D12_DESCRIPTOR_HEAP_DESC desc = heap->GetDesc();
    heap_type = desc.Type;
    heap_start_cpu = heap->GetCPUDescriptorHandleForHeapStart();
    heap_start_gpu = heap->GetGPUDescriptorHandleForHeapStart();
    heap_handle_increment = device->GetDescriptorHandleIncrementSize(heap_type);
    free_indices.reserve((int)desc.NumDescriptors);
    for (int n = desc.NumDescriptors; n > 0; n--)
      free_indices.push_back(n - 1);
  }

  void Destroy() {
    heap = nullptr;
    free_indices.clear();
  }

  void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE *out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE *out_gpu_desc_handle) {
    IM_ASSERT(free_indices.Size > 0);
    int idx = free_indices.back();
    free_indices.pop_back();
    out_cpu_desc_handle->ptr = heap_start_cpu.ptr + (idx * heap_handle_increment);
    out_gpu_desc_handle->ptr = heap_start_gpu.ptr + (idx * heap_handle_increment);
  }
  void Free(D3D12_CPU_DESCRIPTOR_HANDLE out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE out_gpu_desc_handle) {
    int cpu_idx = (int)((out_cpu_desc_handle.ptr - heap_start_cpu.ptr) / heap_handle_increment);
    int gpu_idx = (int)((out_gpu_desc_handle.ptr - heap_start_gpu.ptr) / heap_handle_increment);
    IM_ASSERT(cpu_idx == gpu_idx);
    free_indices.push_back(cpu_idx);
  }
};

struct D3D12ImGuiAssets {
  ImGuiContext *context;

  std::unique_ptr<d3d12::DescriptorHeap> srv_heap;
  ExampleDescriptorHeapAllocator descriptor_alloc;

  DXGI_FORMAT rtv_format;

  std::string font_path;
  float font_size;
  bool draw_command;
};
}  // namespace grassland::graphics::backend
