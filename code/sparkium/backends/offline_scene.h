#pragma once
// Flattening of a live Sparkium scene into the portable DeviceScene that the
// CPU and CUDA backends shade.
//
// The online backends keep one byte-address buffer per mesh and material and
// dispatch HLSL through a graphics API. The offline backends download exactly
// those buffers and rebuild the same layouts in flat host arrays, so the
// shading core in backends/core can run on the CPU and inside CUDA kernels
// without a graphics dispatch. Every layout mirrors the HLSL one:
//   * meshes      -> GeometryHeader + vertex arrays (byte-identical blob)
//   * instances   -> SoftwareInstance (object_to_world/world_to_object/mesh/material)
//   * materials   -> the per-material sampler buffers, decoded per kind
//   * lights      -> LightPoint::SamplerPreprocess / LightGeometryMaterial data
//   * light power -> GatherLightPower + GatherPrimitivePower prefix sums
//   * textures    -> linear+repeat RGBA floats, index 0x1000000 | slot for HDR
//   * camera      -> CameraData (resolved from sparkium::Camera, no download)

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sparkium/backends/core/structs.h"
#include "sparkium/backends/sobol_rows.h"

namespace sparkium {
class Camera;
class Scene;
}  // namespace sparkium

namespace sparkium::backends {

class OfflineScene {
 public:
  // Flattens `scene` for `camera`. Throws std::runtime_error for components the
  // offline backends cannot express (shader-graph materials, hair geometry,
  // non-invertible instance transforms).
  static std::unique_ptr<OfflineScene> Build(sparkium::Scene *scene, sparkium::Camera *camera);

  const DeviceScene &Device() const {
    return device_;
  }

  // Sobol rows are generated lazily: the online table has 65536 rows, but the
  // offline backends only read `accumulated_samples + samples_per_dispatch`.
  const uint32_t *SobolRows() const {
    return sobol_rows_.empty() ? nullptr : sobol_rows_.data();
  }
  uint32_t SobolRowCount() const {
    return static_cast<uint32_t>(sobol_rows_.size() / kSobolDimensions);
  }
  // Regenerates the table when it is shorter than `rows` and refreshes the
  // device pointers.
  bool EnsureSobolRows(uint32_t rows);

  std::string Description() const;

  // Accessors used by the CUDA backend to mirror the arrays into device memory.
  const std::vector<uint8_t> &MeshData() const {
    return mesh_data_;
  }
  const std::vector<MeshRange> &Meshes() const {
    return meshes_;
  }
  const std::vector<SoftwareNode> &MeshNodes() const {
    return mesh_nodes_;
  }
  const std::vector<SoftwareNode> &InstanceNodes() const {
    return instance_nodes_;
  }
  const std::vector<InstanceData> &Instances() const {
    return instances_;
  }
  const std::vector<MaterialData> &Materials() const {
    return materials_;
  }
  const std::vector<LightData> &Lights() const {
    return lights_;
  }
  const std::vector<float> &LightPowerCdf() const {
    return light_power_cdf_;
  }
  const std::vector<float> &PrimitivePowerCdf() const {
    return primitive_power_cdf_;
  }
  const std::vector<TextureData> &Textures() const {
    return textures_;
  }
  const std::vector<float> &TexturePixels() const {
    return texture_pixels_;
  }
  const std::vector<uint32_t> &SobolTable() const {
    return sobol_rows_;
  }

 private:
  OfflineScene() = default;
  void RefreshDevicePointers();

  DeviceScene device_{};
  std::vector<uint8_t> mesh_data_;
  std::vector<MeshRange> meshes_;
  std::vector<SoftwareNode> mesh_nodes_;
  std::vector<SoftwareNode> instance_nodes_;
  std::vector<InstanceData> instances_;
  std::vector<MaterialData> materials_;
  std::vector<LightData> lights_;
  std::vector<float> light_power_cdf_;
  std::vector<float> primitive_power_cdf_;
  std::vector<TextureData> textures_;
  std::vector<float> texture_pixels_;
  std::vector<uint32_t> sobol_rows_;
  float camera_scale_[2]{};
  float aperture_[5]{};
  float camera_to_world_[16]{};
};

// Cheap identity of the scene inputs the flattening depends on. The CUDA
// backend uses it to decide whether its device copies need a refresh.
std::vector<uint64_t> OfflineSceneSignature(sparkium::Scene *scene, sparkium::Camera *camera);

}  // namespace sparkium::backends
