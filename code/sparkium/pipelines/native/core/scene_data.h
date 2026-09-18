#pragma once

// Host-side scene flattening for the native CPU/CUDA backends.
//
// `raytracing::Scene::UpdatePipeline` registers buffers, images and lights in
// a fixed order and uploads them with explicit byte layouts. This class walks
// the very same `sparkium::Scene` in the very same order and produces those
// bytes in host memory, so `pipelines/native/shared/*` -- the shading core
// shared by both native backends -- reads identical offsets and identical
// values.

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "sparkium/core/camera.h"
#include "sparkium/core/film.h"
#include "sparkium/core/scene.h"
#include "sparkium/entity/entities.h"
#include "sparkium/geometry/geometries.h"
#include "sparkium/material/materials.h"
#include "sparkium/pipelines/native/core/bvh_builder.h"
#include "sparkium/pipelines/native/shared/native_integrator.h"

namespace sparkium::native {

// A flattened byte-address buffer. `key` identifies the object the bytes were
// built from and `revision` only advances when they change, so a device mirror
// can skip untouched uploads.
struct DataBuffer {
  const void *key{nullptr};
  uint64_t revision{0};
  const std::vector<uint32_t> *data{nullptr};
};

// A texture downloaded from its `graphics::Image`. Exactly one of the two
// pixel vectors is populated, matching `DeviceTexture`.
struct HostTexture {
  uint64_t revision{0};
  int width{0};
  int height{0};
  std::vector<uint32_t> sdr;
  std::vector<float> hdr;
};

// A compiled shader graph with its payload arrays.
struct HostGraphProgram {
  std::vector<GraphInstruction> instructions;
  std::vector<float4> constants;
  std::vector<float> data;
  int32_t outputs[GRAPH_SURFACE_OUTPUT_COUNT];
};

class SceneData {
 public:
  explicit SceneData(sparkium::Core *core);

  // Rebuilds the flattened scene for one dispatch.
  void Update(sparkium::Scene *scene, sparkium::Camera *camera, const sparkium::Film::Info &film_info);

  // Device-side view backed by the host storage; valid until the next Update.
  const SceneView &View() const {
    return view_;
  }

  const std::vector<DataBuffer> &DataBuffers() const {
    return data_buffers_;
  }
  const std::vector<const HostTexture *> &SdrTextures() const {
    return sdr_textures_;
  }
  const std::vector<const HostTexture *> &HdrTextures() const {
    return hdr_textures_;
  }
  const std::vector<NativeMaterial> &Materials() const {
    return materials_;
  }
  const std::vector<HostGraphProgram> &GraphPrograms() const {
    return graph_programs_;
  }
  const std::vector<uint32_t> &Nodes() const {
    return bvh_.Nodes();
  }
  const std::vector<uint32_t> &Instances() const {
    return instance_records_;
  }
  const std::vector<uint32_t> &Sobol() const {
    return sobol_;
  }
  // Advances whenever the BVH or the instance records were rebuilt.
  uint64_t GeometryRevision() const {
    return geometry_revision_;
  }

 private:
  struct CachedBlob {
    uint64_t revision{0};
    std::vector<uint32_t> data;
  };

  void UpdateGeometryMaterial(sparkium::EntityGeometryMaterial *entity);
  void UpdatePointLight(sparkium::EntityPointLight *entity);

  // Mirrors `Scene::RegisterBuffer`: the first registration assigns the index,
  // later ones reuse it.
  int32_t RegisterBuffer(const void *key, std::vector<uint32_t> &&data);
  // Mirrors `Scene::RegisterImage`, including the 0x1000000 HDR tag.
  int32_t RegisterImage(graphics::Image *image);
  int32_t RegisterMaterial(sparkium::Material *material);

  // Blob builders reproducing the per-object uploads of the graphics pipeline.
  std::vector<uint32_t> GeometryBlob(sparkium::GeometryMesh *geometry);
  std::vector<uint32_t> MaterialBlob(sparkium::Material *material);
  std::vector<uint32_t> MeshLightBlob(uint32_t primitive_count,
                                      uint32_t material_type,
                                      const std::vector<uint32_t> &geometry_blob,
                                      const std::vector<uint32_t> &material_blob,
                                      const glm::mat4x3 &transform);
  int32_t MeshLightSamplerShader(sparkium::Material *material) const;

  void BuildLightSelector(uint32_t light_count);
  void BuildInstanceRecords();
  void BuildView(const sparkium::Scene::Settings::RayTracing &settings, const sparkium::Film::Info &film_info);

  sparkium::Core *core_;
  SceneView view_{};
  BvhBuilder bvh_;

  // Blob storage keyed by the source object, so unchanged bytes keep their
  // revision across frames.
  std::map<const void *, CachedBlob> blob_cache_;
  std::map<graphics::Image *, HostTexture> texture_cache_;
  uint64_t next_revision_{1};
  uint64_t geometry_revision_{0};

  std::vector<DataBuffer> data_buffers_;
  std::map<const void *, int32_t> buffer_index_;
  std::vector<const HostTexture *> sdr_textures_;
  std::vector<const HostTexture *> hdr_textures_;
  std::map<graphics::Image *, int32_t> sdr_index_;
  std::map<graphics::Image *, int32_t> hdr_index_;
  std::vector<NativeMaterial> materials_;
  std::vector<HostGraphProgram> graph_programs_;
  std::map<sparkium::Material *, int32_t> material_index_;

  std::vector<uint32_t> instance_metadatas_;  // 3 uint32 per instance.
  std::vector<uint32_t> light_metadatas_;     // 4 uint32 per light.
  std::vector<uint32_t> light_selector_;      // count, then the power CDF.
  std::vector<uint32_t> camera_data_;
  std::vector<uint32_t> instance_records_;    // 16-byte header + 112 bytes each.
  std::vector<uint32_t> sobol_;

  std::vector<BvhInstance> instances_;
  std::vector<sparkium::GeometryMesh *> instance_geometries_;

  // Pointer arrays referenced by `view_`.
  std::vector<ByteBuffer> view_data_buffers_;
  std::vector<DeviceTexture> view_sdr_textures_;
  std::vector<DeviceTexture> view_hdr_textures_;
  std::vector<GraphProgram> view_graph_programs_;
};

}  // namespace sparkium::native
