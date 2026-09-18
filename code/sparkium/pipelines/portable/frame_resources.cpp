#include "sparkium/pipelines/portable/frame_resources.h"

#include <cstring>

namespace sparkium::portable {

void PopulateContext(const BakeResult &bake,
                     const std::vector<uint32_t> &sobol_table,
                     sparkium_portable::float4 *color,
                     float *samples,
                     uint32_t width,
                     uint32_t height,
                     FrameResources &res) {
  using namespace sparkium_portable;
  res.table.clear();
  std::vector<size_t> sizes;
  auto add_buffer = [&res, &sizes](const std::vector<uint8_t> &buffer) {
    res.table.push_back(buffer.empty() ? nullptr : buffer.data());
    sizes.push_back(buffer.size());
  };
  for (const auto &geometry : bake.geometries)
    add_buffer(geometry.buffer);
  for (const auto &material : bake.material_buffers)
    add_buffer(material.buffer);
  for (const auto &light : bake.lights)
    add_buffer(light.buffer);
  const size_t count = res.table.size();
  res.table.resize(count + (count * sizeof(size_t) + sizeof(const uint8_t *) - 1) / sizeof(const uint8_t *));
  std::memcpy(res.table.data() + count, sizes.data(), count * sizeof(size_t));

  res.instance_metadatas.resize(std::max<size_t>(1, bake.instance_metadatas.size()) * sizeof(GpuInstanceMetadata));
  if (!bake.instance_metadatas.empty())
    std::memcpy(res.instance_metadatas.data(), bake.instance_metadatas.data(),
                bake.instance_metadatas.size() * sizeof(GpuInstanceMetadata));
  res.light_metadatas.resize(std::max<size_t>(1, bake.lights.size()) * sizeof(GpuLightMetadata));
  for (size_t i = 0; i < bake.lights.size(); ++i)
    std::memcpy(res.light_metadatas.data() + i * sizeof(GpuLightMetadata), &bake.lights[i].metadata,
                sizeof(GpuLightMetadata));
  res.light_selector.resize(sizeof(uint32_t) + std::max<size_t>(1, bake.light_power_cdf.size()) * sizeof(float));
  {
    const uint32_t light_count = static_cast<uint32_t>(bake.lights.size());
    std::memcpy(res.light_selector.data(), &light_count, 4);
    if (!bake.light_power_cdf.empty())
      std::memcpy(res.light_selector.data() + 4, bake.light_power_cdf.data(),
                  bake.light_power_cdf.size() * sizeof(float));
  }
  res.instances.resize(std::max<size_t>(1, bake.instances.size()) * sizeof(Instance) + 16);
  {
    const uint32_t instance_count = static_cast<uint32_t>(bake.instances.size());
    std::memset(res.instances.data(), 0, 16);
    std::memcpy(res.instances.data(), &instance_count, 4);
    if (!bake.instances.empty())
      std::memcpy(res.instances.data() + 16, bake.instances.data(), bake.instances.size() * sizeof(Instance));
  }
  res.sdr_textures.resize(bake.sdr_textures.size());
  for (size_t i = 0; i < bake.sdr_textures.size(); ++i)
    res.sdr_textures[i] = bake.sdr_textures[i].texture;
  res.hdr_textures.resize(bake.hdr_textures.size());
  for (size_t i = 0; i < bake.hdr_textures.size(); ++i)
    res.hdr_textures[i] = bake.hdr_textures[i].texture;

  // The Sobol asset stores direction numbers bit-cast as floats; the kernel
  // reads them as uints.
  res.sobol_bits.resize(sobol_table.size());
  std::memcpy(res.sobol_bits.data(), sobol_table.data(), sobol_table.size() * sizeof(uint32_t));

  KernelContext &ctx = res.ctx;
  ctx = KernelContext{};
  ctx.accumulated_color = color;
  ctx.accumulated_samples = samples;
  ctx.image_width = width;
  ctx.image_height = height;
  ctx.render_settings = bake.render_settings.data();
  ctx.sobol_table = res.sobol_bits.empty() ? nullptr : res.sobol_bits.data();
  ctx.camera_data = bake.camera_data.data();
  ctx.data_buffers = count ? reinterpret_cast<const uint8_t *const *>(res.table.data()) : nullptr;
  ctx.data_buffer_count = static_cast<uint32_t>(count);
  ctx.instance_metadatas = res.instance_metadatas.data();
  ctx.light_selector_data = res.light_selector.data();
  ctx.light_metadatas = res.light_metadatas.data();
  ctx.software_instances = res.instances.data();
  ctx.software_nodes = bake.nodes.empty() ? nullptr : reinterpret_cast<const uint8_t *>(bake.nodes.data());
  ctx.sdr_textures = res.sdr_textures.empty() ? nullptr : res.sdr_textures.data();
  ctx.sdr_texture_count = static_cast<uint32_t>(res.sdr_textures.size());
  ctx.hdr_textures = res.hdr_textures.empty() ? nullptr : res.hdr_textures.data();
  ctx.hdr_texture_count = static_cast<uint32_t>(res.hdr_textures.size());
}

}  // namespace sparkium::portable
