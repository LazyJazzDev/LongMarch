#include <atomic>
#include <thread>
#include <vector>

#include "sparkium/pipelines/portable/frame_resources.h"

namespace sparkium_portable {
// Host kernels compiled into this binary would use this instance; JIT-loaded
// modules bring their own. Kept for link completeness.
const KernelContext *g_ctx_ptr = nullptr;
}  // namespace sparkium_portable

namespace sparkium::portable {

void RenderPixelsHostPrepared(const BakeResult &bake,
                              const std::vector<uint32_t> &sobol_table,
                              RenderPixelFn entry,
                              void *context_slot,
                              sparkium_portable::float4 *accumulated_color,
                              float *accumulated_samples,
                              uint32_t width,
                              uint32_t height,
                              uint32_t accumulated_sample_base,
                              uint32_t threads) {
  FrameResources res;
  PopulateContext(bake, sobol_table, accumulated_color, accumulated_samples, width, height, res);
  // Patch the sample base: RenderSettings.accumulated_samples is the count
  // accumulated before this frame.
  auto *settings = const_cast<uint8_t *>(res.ctx.render_settings);
  int32_t base = int32_t(accumulated_sample_base);
  std::memcpy(settings + 28, &base, 4);

  // The dlopen'd module owns its own g_ctx_ptr; write through its slot.
  if (context_slot)
    *reinterpret_cast<const sparkium_portable::KernelContext **>(context_slot) = &res.ctx;
  else
    sparkium_portable::g_ctx_ptr = &res.ctx;

  threads = std::max(1u, threads);
  const uint32_t total = width * height;
  std::atomic<uint32_t> next{0};
  auto worker = [&]() {
    for (;;) {
      const uint32_t pixel = next.fetch_add(64, std::memory_order_relaxed);
      if (pixel >= total)
        break;
      const uint32_t end = std::min(pixel + 64, total);
      for (uint32_t i = pixel; i < end; ++i)
        entry(i % width, i / width);
    }
  };
  if (threads <= 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (uint32_t i = 0; i < threads; ++i)
      pool.emplace_back(worker);
    for (auto &thread : pool)
      thread.join();
  }
  if (context_slot)
    *reinterpret_cast<const sparkium_portable::KernelContext **>(context_slot) = nullptr;
  else
    sparkium_portable::g_ctx_ptr = nullptr;
}

}  // namespace sparkium::portable
