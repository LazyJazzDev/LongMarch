#include "sparkium/pipelines/raytracing/core/cpu_pipeline.h"

#include <algorithm>
#include <set>
#include <cstring>
#include <thread>

#include "grassland/util/log.h"

#include "sparkium/pipelines/raytracing/cpu/graph_program.h"
#include "sparkium/pipelines/raytracing/core/geometry.h"
#include "sparkium/pipelines/raytracing/core/material.h"
#include "sparkium/pipelines/raytracing/material/materials.h"

namespace sparkium::raytracing {
namespace {

// A triangle's bounds, read straight out of the geometry buffer using the same
// offsets geometry/mesh/hit_record.hlsli uses.
bool ReadTriangle(const uint8_t *geometry,
                  size_t geometry_size,
                  uint32_t primitive,
                  float lo[3],
                  float hi[3]) {
  auto load_u32 = [&](size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, geometry + offset, sizeof(value));
    return value;
  };
  const size_t position_offset = load_u32(8);
  const size_t position_stride = load_u32(12);
  const size_t index_offset = load_u32(48);

  const size_t index_base = index_offset + static_cast<size_t>(primitive) * 3 * sizeof(uint32_t);
  if (index_base + 3 * sizeof(uint32_t) > geometry_size)
    return false;
  uint32_t ids[3];
  std::memcpy(ids, geometry + index_base, sizeof(ids));

  for (int axis = 0; axis < 3; ++axis) {
    lo[axis] = std::numeric_limits<float>::max();
    hi[axis] = -std::numeric_limits<float>::max();
  }
  for (int vertex = 0; vertex < 3; ++vertex) {
    const size_t offset = position_offset + position_stride * ids[vertex];
    if (offset + sizeof(float) * 3 > geometry_size)
      return false;
    float position[3];
    std::memcpy(position, geometry + offset, sizeof(position));
    for (int axis = 0; axis < 3; ++axis) {
      lo[axis] = std::min(lo[axis], position[axis]);
      hi[axis] = std::max(hi[axis], position[axis]);
    }
  }
  return true;
}

// The four built-in samplers plus the shader-graph one. Mirrors how
// LightGeometryMaterial picks its evaluator, and what SoftwarePipeline's
// generated dispatch distinguishes by source text.
uint32_t KernelOf(Material *material) {
  if (dynamic_cast<MaterialLambertian *>(material))
    return cpu::kMaterialKernelLambertian;
  if (dynamic_cast<MaterialLight *>(material))
    return cpu::kMaterialKernelLight;
  if (dynamic_cast<MaterialPrincipled *>(material))
    return cpu::kMaterialKernelPrincipled;
  if (dynamic_cast<MaterialSpecular *>(material))
    return cpu::kMaterialKernelSpecular;
  if (dynamic_cast<MaterialShaderGraph *>(material))
    return cpu::kMaterialKernelShaderGraph;
  return cpu::kMaterialKernelLambertian;
}

}  // namespace

CpuPipeline::CpuPipeline(Core *core) : core_(core) {
}

CpuPipeline::~CpuPipeline() = default;

void CpuPipeline::ClearInstances() {
  instances_.clear();
}

void CpuPipeline::AddInstance(Geometry *geometry,
                              Material *material,
                              const glm::mat4x3 &transform,
                              uint32_t geometry_index) {
  instances_.push_back({geometry, material, transform, geometry_index});
}

void CpuPipeline::SetThreadCount(uint32_t thread_count) {
  thread_count_ = thread_count;
}

void CpuPipeline::SetGraphEngine(cpu::GraphEngine engine) {
  graph_engine_ = engine;
}

void CpuPipeline::ResetFilm() {
  std::fill(accumulated_color_.begin(), accumulated_color_.end(), 0.0f);
  std::fill(accumulated_samples_.begin(), accumulated_samples_.end(), 0.0f);
}

void CpuPipeline::RefreshBuffers(const std::vector<graphics::Buffer *> &scene_buffers,
                                 graphics::Buffer *camera_buffer,
                                 graphics::Buffer *sobol_buffer,
                                 graphics::Buffer *instance_metadata_buffer,
                                 graphics::Buffer *light_selector_buffer,
                                 graphics::Buffer *light_metadata_buffer) {
  // Slot order matches Scene::Render's binding for the GPU paths: the scene's
  // buffers first, then the fixed slots the shader's index macros reach.
  std::vector<graphics::Buffer *> sources = scene_buffers;
  sources.push_back(sobol_buffer);
  sources.push_back(camera_buffer);
  sources.push_back(instance_metadata_buffer);
  sources.push_back(light_selector_buffer);
  sources.push_back(light_metadata_buffer);

  buffers_.resize(sources.size());
  for (size_t i = 0; i < sources.size(); ++i) {
    HostBuffer &host = buffers_[i];
    host.source = sources[i];
    host.bytes.resize(sources[i] ? sources[i]->Size() : 0);
    if (sources[i] && !host.bytes.empty())
      sources[i]->DownloadData(host.bytes.data(), host.bytes.size());
  }

  data_buffer_views_.resize(sources.size() + 1);
  for (size_t i = 0; i < sources.size(); ++i)
    data_buffer_views_[i] = cpu::BufferView{buffers_[i].bytes.data(), buffers_[i].bytes.size()};
  // The software instance array is the backend's own; it is the last slot.
  data_buffer_views_.back() = cpu::BufferView{instance_bytes_.data(), instance_bytes_.size()};
}

const uint8_t *CpuPipeline::BufferBytes(graphics::Buffer *buffer) const {
  for (const HostBuffer &host : buffers_)
    if (host.source == buffer)
      return host.bytes.data();
  return nullptr;
}

bool CpuPipeline::EnsureGeometryTree(size_t instance_index) {
  Geometry *geometry = instances_[instance_index].geometry;
  if (std::any_of(geometry_trees_.begin(), geometry_trees_.end(),
                  [geometry](const GeometryTree &tree) { return tree.geometry == geometry; }))
    return true;

  graphics::Buffer *source = geometry->Buffer();
  const uint8_t *bytes = BufferBytes(source);
  const size_t size = source ? source->Size() : 0;
  if (!bytes || size == 0)
    return false;

  const uint32_t primitive_count = static_cast<uint32_t>(geometry->PrimitiveCount());
  bvh_scratch_.resize(primitive_count);
  for (uint32_t i = 0; i < primitive_count; ++i) {
    float lo[3], hi[3];
    if (!ReadTriangle(bytes, size, i, lo, hi))
      return false;
    std::copy(lo, lo + 3, bvh_scratch_[i].lo);
    std::copy(hi, hi + 3, bvh_scratch_[i].hi);
    bvh_scratch_[i].index = i;
  }

  // Mesh trees follow the top-level tree, which always occupies the first
  // BvhNodeCount(instance_count) slots.
  GeometryTree tree;
  tree.geometry = geometry;
  tree.root = static_cast<uint32_t>(nodes_.size());
  tree.primitive_count = primitive_count;
  tree.buffer_index = instances_[instance_index].geometry_index;

  nodes_.resize(nodes_.size() + cpu::BvhNodeCount(primitive_count));
  cpu::BuildBvh(nodes_, tree.root, bvh_scratch_);
  geometry_trees_.push_back(tree);
  return true;
}

void CpuPipeline::BuildAccelerationStructures() {
  const uint32_t instance_count = static_cast<uint32_t>(instances_.size());
  const size_t tlas_nodes = cpu::BvhNodeCount(instance_count);

  // Materials are compacted by object identity: each has its own parameter
  // buffer and sampler, so identity is what the shader's per-material dispatch
  // needs. SoftwarePipeline compacts by source text instead because it has to
  // generate one kernel per distinct source.
  material_order_.clear();
  material_kernels_.clear();
  std::vector<uint32_t> instance_material(instance_count, 0);
  for (uint32_t i = 0; i < instance_count; ++i) {
    Material *material = instances_[i].material;
    auto found = std::find(material_order_.begin(), material_order_.end(), material);
    if (found == material_order_.end()) {
      instance_material[i] = static_cast<uint32_t>(material_order_.size());
      material_order_.push_back(material);
      material_kernels_.push_back(KernelOf(material));
    } else {
      instance_material[i] = static_cast<uint32_t>(found - material_order_.begin());
    }
  }

  // Cached mesh trees store absolute node indices, and the top-level tree's
  // size depends on the instance count, so the cache only survives a change in
  // instance count that keeps that size the same.
  if (tlas_nodes != cached_tlas_nodes_) {
    geometry_trees_.clear();
    cached_tlas_nodes_ = tlas_nodes;
  }
  geometry_trees_.erase(std::remove_if(geometry_trees_.begin(), geometry_trees_.end(),
                                       [&](const GeometryTree &tree) {
                                         return std::none_of(instances_.begin(), instances_.end(),
                                                             [&](const Instance &instance) {
                                                               return instance.geometry == tree.geometry;
                                                             });
                                       }),
                        geometry_trees_.end());

  nodes_.clear();
  nodes_.resize(tlas_nodes);
  for (size_t i = 0; i < instances_.size(); ++i)
    EnsureGeometryTree(i);

  // The instance's world bounds come from transforming its mesh tree's root
  // box, which is what software/build.hlsl does when it writes top-level
  // leaves.
  std::vector<cpu::BvhPrimitive> tlas_primitives(instance_count);
  for (uint32_t i = 0; i < instance_count; ++i) {
    const auto tree = std::find_if(geometry_trees_.begin(), geometry_trees_.end(),
                                   [&](const GeometryTree &candidate) {
                                     return candidate.geometry == instances_[i].geometry;
                                   });
    float lo[3] = {0.0f, 0.0f, 0.0f};
    float hi[3] = {0.0f, 0.0f, 0.0f};
    if (tree != geometry_trees_.end()) {
      const cpu::SoftwareNode &root = nodes_[tree->root];
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::numeric_limits<float>::max();
        hi[axis] = -std::numeric_limits<float>::max();
      }
      for (int corner = 0; corner < 8; ++corner) {
        const glm::vec4 local((corner & 1) ? root.hi[0] : root.lo[0], (corner & 2) ? root.hi[1] : root.lo[1],
                              (corner & 4) ? root.hi[2] : root.lo[2], 1.0f);
        // glm::mat4x3 is HLSL's float3x4: four columns of three, so it
        // transforms a homogeneous point into a three component result.
        const glm::vec3 world = instances_[i].transform * local;
        for (int axis = 0; axis < 3; ++axis) {
          lo[axis] = std::min(lo[axis], world[axis]);
          hi[axis] = std::max(hi[axis], world[axis]);
        }
      }
    }
    std::copy(lo, lo + 3, tlas_primitives[i].lo);
    std::copy(hi, hi + 3, tlas_primitives[i].hi);
    tlas_primitives[i].index = i;
  }
  if (instance_count > 0)
    cpu::BuildBvh(nodes_, 0, tlas_primitives);

  // SoftwareInstance records, in software/layout.hlsli's layout: the two
  // transforms, then root/geometry/material/primitive_count.
  instance_bytes_.assign(16 + static_cast<size_t>(instance_count) * 112, 0);
  std::memcpy(instance_bytes_.data(), &instance_count, sizeof(instance_count));
  for (uint32_t i = 0; i < instance_count; ++i) {
    uint8_t *record = instance_bytes_.data() + 16 + static_cast<size_t>(i) * 112;
    const glm::mat4 object_to_world(instances_[i].transform);
    const glm::mat4 world_to_object = glm::inverse(object_to_world);
    // glm::mat4x3 is column major: four columns of three floats, the fourth
    // holding the translation, which is what LoadFloat3x4 reads and transposes
    // back into a float3x4. Taking the first twelve floats of the mat4 instead
    // would silently drop the translation.
    const glm::mat4x3 object_to_world_affine(object_to_world);
    const glm::mat4x3 world_to_object_affine(world_to_object);
    std::memcpy(record, &object_to_world_affine[0][0], sizeof(float) * 12);
    std::memcpy(record + 48, &world_to_object_affine[0][0], sizeof(float) * 12);
    uint32_t *info = reinterpret_cast<uint32_t *>(record + 96);
    const auto tree = std::find_if(geometry_trees_.begin(), geometry_trees_.end(),
                                   [&](const GeometryTree &candidate) {
                                     return candidate.geometry == instances_[i].geometry;
                                   });
    info[0] = tree == geometry_trees_.end() ? 0 : tree->root;
    info[1] = instances_[i].geometry_index;
    info[2] = instance_material[i];
    info[3] = tree == geometry_trees_.end() ? 0 : tree->primitive_count;
  }
}

void CpuPipeline::RefreshGraphPrograms() {
  // Graph materials are the only ones whose behaviour is not compiled in, so
  // their generated source is handed to an engine here. Recompiling every frame
  // would be wasteful, so this only runs when the set of graph materials
  // changes.
  std::vector<Material *> graphs;
  for (const Instance &instance : instances_) {
    if (instance.material && instance.material->GraphImpl())
      graphs.push_back(instance.material);
  }
  std::sort(graphs.begin(), graphs.end());
  graphs.erase(std::unique(graphs.begin(), graphs.end()), graphs.end());
  if (graphs == graph_materials_)
    return;
  graph_materials_ = graphs;

  cpu::ClearGraphMaterials();
  if (graphs.empty())
    return;

  // The shader looks a material up by the buffer slot recorded in its instance
  // metadata, so that is the key the programs are registered under. Several
  // instances can share a material, hence the dedup.
  const cpu::BufferView &metadata = data_buffer_views_[data_buffer_views_.size() - 4];
  std::set<int32_t> registered;
  for (size_t i = 0; i < instances_.size(); ++i) {
    Material *material = instances_[i].material;
    const CodeLines *source = material ? material->GraphImpl() : nullptr;
    if (!source)
      continue;
    if (!metadata.data || metadata.size < (i + 1) * 12)
      continue;
    int32_t material_data_index = 0;
    std::memcpy(&material_data_index, metadata.data + i * 12 + 4, sizeof(material_data_index));
    if (material_data_index < 0 || !registered.insert(material_data_index).second)
      continue;
    cpu::RegisterGraphMaterial(static_cast<uint32_t>(material_data_index), static_cast<std::string>(*source));
  }
  if (!cpu::FinalizeGraphMaterials(graph_engine_, sdr_texture_views_, hdr_texture_views_))
    LogError("[sparkium] the CPU backend could not prepare this scene's shader graphs");
}

void CpuPipeline::GatherLightPowers(graphics::Buffer *light_selector_buffer,
                                    graphics::Buffer *light_metadata_buffer) {
  // The host equivalent of gather_light_power.hlsl: an inclusive prefix sum of
  // each light's power, stored after the count.
  if (!light_metadata_buffer || light_metadata_buffer->Size() < sizeof(LightMetadata))
    return;
  const uint32_t light_count = static_cast<uint32_t>(light_metadata_buffer->Size() / sizeof(LightMetadata));
  std::vector<LightMetadata> metadatas(light_count);
  light_metadata_buffer->DownloadData(metadatas.data(), light_count * sizeof(LightMetadata));

  std::vector<float> cdf(light_count);
  float running = 0.0f;
  for (uint32_t i = 0; i < light_count; ++i) {
    float power = 0.0f;
    if (metadatas[i].sampler_data_index >= 0 &&
        static_cast<size_t>(metadatas[i].sampler_data_index) < data_buffer_views_.size()) {
      const cpu::BufferView &view = data_buffer_views_[metadatas[i].sampler_data_index];
      if (view.data && metadatas[i].power_offset + sizeof(float) <= view.size)
        std::memcpy(&power, view.data + metadatas[i].power_offset, sizeof(power));
    }
    running += power;
    cdf[i] = running;
  }
  if (cdf.empty())
    return;
  // The shader reads the host copy of the selector buffer, and it was taken
  // before the CDF existed, so the CDF is written there and then pushed to the
  // buffer itself.
  light_selector_buffer->UploadData(cdf.data(), cdf.size() * sizeof(float), 4);
  for (HostBuffer &host : buffers_) {
    if (host.source == light_selector_buffer && host.bytes.size() >= 4 + cdf.size() * sizeof(float))
      std::memcpy(host.bytes.data() + 4, cdf.data(), cdf.size() * sizeof(float));
  }
}

void CpuPipeline::RefreshTextures(const std::vector<graphics::Image *> &sdr_images,
                                  const std::vector<graphics::Image *> &hdr_images) {
  auto convert = [](graphics::Image *image, std::vector<float> &pixels, cpu::TextureView &view) {
    const graphics::Extent2D extent = image->Extent();
    const size_t count = static_cast<size_t>(extent.width) * extent.height;
    pixels.assign(count * 4, 0.0f);
    if (image->Format() == graphics::IMAGE_FORMAT_R8G8B8A8_UNORM) {
      std::vector<uint8_t> raw(count * 4);
      image->DownloadData(raw.data());
      for (size_t i = 0; i < count; ++i)
        for (int channel = 0; channel < 4; ++channel)
          pixels[i * 4 + channel] = static_cast<float>(raw[i * 4 + channel]) / 255.0f;
    } else if (image->Format() == graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT) {
      image->DownloadData(pixels.data());
    }
    view = cpu::TextureView{pixels.data(), extent.width, extent.height, 4};
  };

  sdr_texture_pixels_.resize(sdr_images.size());
  sdr_texture_views_.resize(sdr_images.size());
  for (size_t i = 0; i < sdr_images.size(); ++i)
    convert(sdr_images[i], sdr_texture_pixels_[i], sdr_texture_views_[i]);

  hdr_texture_pixels_.resize(hdr_images.size());
  hdr_texture_views_.resize(hdr_images.size());
  for (size_t i = 0; i < hdr_images.size(); ++i)
    convert(hdr_images[i], hdr_texture_pixels_[i], hdr_texture_views_[i]);
}

void CpuPipeline::Update(const std::vector<graphics::Buffer *> &scene_buffers,
                         graphics::Buffer *camera_buffer,
                         graphics::Buffer *sobol_buffer,
                         graphics::Buffer *instance_metadata_buffer,
                         graphics::Buffer *light_selector_buffer,
                         graphics::Buffer *light_metadata_buffer,
                         const std::vector<graphics::Image *> &sdr_images,
                         const std::vector<graphics::Image *> &hdr_images) {
  RefreshBuffers(scene_buffers, camera_buffer, sobol_buffer, instance_metadata_buffer, light_selector_buffer,
                 light_metadata_buffer);
  BuildAccelerationStructures();
  RefreshGraphPrograms();
  data_buffer_views_.back() = cpu::BufferView{instance_bytes_.data(), instance_bytes_.size()};
  // The light CDF is read out of the scene buffers, so it has to run after the
  // refreshed views are in place.
  GatherLightPowers(light_selector_buffer, light_metadata_buffer);
  RefreshTextures(sdr_images, hdr_images);
}

void CpuPipeline::Render(sparkium::Film *film, const sparkium::Scene::Settings::RayTracing &settings) {
  const uint32_t width = static_cast<uint32_t>(film->GetWidth());
  const uint32_t height = static_cast<uint32_t>(film->GetHeight());
  if (film_width_ != width || film_height_ != height) {
    film_width_ = width;
    film_height_ = height;
    accumulated_color_.assign(static_cast<size_t>(width) * height * 4, 0.0f);
    accumulated_samples_.assign(static_cast<size_t>(width) * height, 0.0f);
  }

  cpu::FrameBindings bindings;
  bindings.nodes = cpu::BufferView{reinterpret_cast<const uint8_t *>(nodes_.data()),
                                   nodes_.size() * sizeof(cpu::SoftwareNode)};
  bindings.data_buffers = data_buffer_views_;
  bindings.sdr_textures = sdr_texture_views_;
  bindings.hdr_textures = hdr_texture_views_;
  bindings.accumulated_color = cpu::TextureView{accumulated_color_.data(), width, height, 4};
  bindings.accumulated_samples = cpu::TextureView{accumulated_samples_.data(), width, height, 1};
  bindings.width = width;
  bindings.height = height;
  bindings.material_kernels = material_kernels_;
  bindings.settings.samples_per_dispatch = settings.samples_per_dispatch;
  bindings.settings.max_bounces = settings.max_bounces;
  bindings.settings.alpha_shadow = settings.alpha_shadow ? 1 : 0;
  bindings.settings.background_color[0] = settings.background_color.r;
  bindings.settings.background_color[1] = settings.background_color.g;
  bindings.settings.background_color[2] = settings.background_color.b;
  bindings.settings.accumulated_samples = film->info.accumulated_samples;
  bindings.settings.persistence = film->info.persistence;
  bindings.settings.clamping = film->info.clamping;
  bindings.settings.max_exposure = film->info.max_exposure;
  bindings.settings.view_transform = film->info.view_transform;
  bindings.settings.exposure = film->info.exposure;
  bindings.settings.gamma = film->info.gamma;
  bindings.settings.contrast = film->info.contrast;

  uint32_t threads = thread_count_;
  if (threads == 0)
    threads = std::max(1u, std::thread::hardware_concurrency());

  cpu::BindFrame(bindings);
  cpu::RenderFrame(bindings, threads);

  film->info.accumulated_samples += settings.samples_per_dispatch;

  // film2img.hlsl averages the accumulation; a pixel with no samples stays
  // black. The result is written straight into the film's HDR image.
  std::vector<float> resolved(static_cast<size_t>(width) * height * 4, 0.0f);
  for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
    const float samples = accumulated_samples_[i];
    if (samples <= 0.0f)
      continue;
    for (int channel = 0; channel < 4; ++channel)
      resolved[i * 4 + channel] = accumulated_color_[i * 4 + channel] / samples;
  }
  film->GetRawImage()->UploadData(resolved.data());
}

}  // namespace sparkium::raytracing
