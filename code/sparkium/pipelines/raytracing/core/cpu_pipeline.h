// The CPU execution path of the raytracing pipeline.
//
// Where SoftwarePipeline builds its acceleration structures and renders with
// compute shaders, this runs the whole thing on the host: it keeps host copies
// of the scene's buffers, builds the BVH with bvh_builder, gathers the light
// CDFs, and dispatches the compiled shaders' RenderDispatch over the film.
//
// It is deliberately constructed the same way SoftwarePipeline is, so Scene can
// drive either one, and it deliberately goes through the host graphics backend
// so that the registration work Scene already does (geometry byte layout,
// material parameter buffers, instance and light metadata) is reused unchanged.
#pragma once

#include "sparkium/pipelines/raytracing/core/core_util.h"
#include "sparkium/pipelines/raytracing/cpu/cpu_shaders.h"
#include "sparkium/pipelines/raytracing/cpu/bvh_builder.h"

namespace sparkium::raytracing {

class CpuPipeline {
 public:
  explicit CpuPipeline(Core *core);
  ~CpuPipeline();

  void ClearInstances();
  void AddInstance(Geometry *geometry, Material *material, const glm::mat4x3 &transform, uint32_t geometry_index);

  // Refreshes every host-side copy and rebuilds the acceleration structures and
  // light CDFs. Called once per frame, after Scene has registered the entities,
  // and takes the same binding set Scene::Render binds for the GPU paths.
  void Update(const std::vector<graphics::Buffer *> &scene_buffers,
              graphics::Buffer *camera_buffer,
              graphics::Buffer *sobol_buffer,
              graphics::Buffer *instance_metadata_buffer,
              graphics::Buffer *light_selector_buffer,
              graphics::Buffer *light_metadata_buffer,
              const std::vector<graphics::Image *> &sdr_images,
              const std::vector<graphics::Image *> &hdr_images);

  // Traces one sample per pixel and resolves the film's raw image.
  void Render(sparkium::Film *film, const sparkium::Scene::Settings::RayTracing &settings);

  // Drops the accumulated film, used by Film::Reset.
  void ResetFilm();

  // Number of worker threads; 0 means one per hardware thread.
  void SetThreadCount(uint32_t thread_count);

  // Which engine evaluates shader graphs. Defaults to the interpreter.
  void SetGraphEngine(cpu::GraphEngine engine);

 private:
  struct Instance {
    Geometry *geometry;
    Material *material;
    glm::mat4x3 transform;
    uint32_t geometry_index;
  };

  // A host copy of one registered buffer, refreshed whenever Update runs.
  struct HostBuffer {
    graphics::Buffer *source{nullptr};
    std::vector<uint8_t> bytes;
  };

  struct GeometryTree {
    Geometry *geometry{nullptr};
    uint32_t root{0};
    uint32_t primitive_count{0};
    uint32_t buffer_index{0};
  };

  void RefreshBuffers(const std::vector<graphics::Buffer *> &scene_buffers,
                      graphics::Buffer *camera_buffer,
                      graphics::Buffer *sobol_buffer,
                      graphics::Buffer *instance_metadata_buffer,
                      graphics::Buffer *light_selector_buffer,
                      graphics::Buffer *light_metadata_buffer);
  const uint8_t *BufferBytes(graphics::Buffer *buffer) const;

  void BuildAccelerationStructures();
  bool EnsureGeometryTree(size_t instance_index);
  void RefreshGraphPrograms();
  void GatherLightPowers(graphics::Buffer *light_selector_buffer, graphics::Buffer *light_metadata_buffer);
  void RefreshTextures(const std::vector<graphics::Image *> &sdr_images,
                       const std::vector<graphics::Image *> &hdr_images);

  Core *core_;
  std::vector<Instance> instances_;
  std::vector<GeometryTree> geometry_trees_;

  std::vector<HostBuffer> buffers_;
  std::vector<cpu::BufferView> data_buffer_views_;
  std::vector<cpu::TextureView> sdr_texture_views_;
  std::vector<cpu::TextureView> hdr_texture_views_;
  std::vector<std::vector<float>> sdr_texture_pixels_;
  std::vector<std::vector<float>> hdr_texture_pixels_;

  std::vector<cpu::SoftwareNode> nodes_;
  std::vector<uint8_t> instance_bytes_;
  std::vector<cpu::BvhPrimitive> bvh_scratch_;
  // Size of the top-level tree the cached mesh tree roots were computed for.
  size_t cached_tlas_nodes_{0};

  // Compacted per material object; index i maps to Instance.material.
  std::vector<uint32_t> material_kernels_;
  std::vector<Material *> material_order_;
  // The graph materials currently compiled; recompiled only when this changes.
  std::vector<Material *> graph_materials_;

  std::vector<float> accumulated_color_;
  std::vector<float> accumulated_samples_;
  uint32_t film_width_{0};
  uint32_t film_height_{0};

  cpu::GraphEngine graph_engine_{cpu::GraphEngine::Interpreter};
  uint32_t thread_count_{0};
};

}  // namespace sparkium::raytracing
