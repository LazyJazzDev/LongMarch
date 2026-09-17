// Regression tests for the offline (CPU/CUDA) Sparkium backends.
//
// The image-level verification of the backends lives in
// scripts/check_offline_backends.py, which renders the bundled scenes with the
// CPU backend, the CUDA backend and the Vulkan rt_fallback reference and
// compares the PNGs. The tests below cover the parts of that pipeline that can
// be asserted exactly:
//   * the lazily generated Sobol rows equal grassland::SobolTableGen,
//   * the texture table addresses each registered image separately,
//   * the flattened mesh blob is byte-identical to the online geometry buffer,
//   * the CPU backend is deterministic and only the CPU performs the work,
//   * CPU and CUDA agree on a directly lit scene, and the CUDA backend reports
//     clearly when it was not built.

#include <gtest/gtest.h>
#include <long_march.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "grassland/util/sobol.h"
#include "sparkium/backends/core/texture.h"
#include "sparkium/backends/offline_backend.h"
#include "../../demo/sparkium_backend.h"

using namespace grassland;

namespace {

std::vector<uint8_t> DownloadBuffer(graphics::Buffer *buffer) {
  std::vector<uint8_t> data(buffer->Size());
  buffer->DownloadData(data.data(), data.size());
  return data;
}

// Two constant images with distinct colors; the second one is only reachable
// through its registered index.
std::unique_ptr<graphics::Image> ConstantImage(graphics::Core *graphics, uint8_t r, uint8_t g, uint8_t b) {
  std::unique_ptr<graphics::Image> image;
  EXPECT_EQ(graphics->CreateImage(4, 4, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image), 0);
  std::vector<uint8_t> pixels(4 * 4 * 4);
  for (size_t i = 0; i < pixels.size(); i += 4) {
    pixels[i + 0] = r;
    pixels[i + 1] = g;
    pixels[i + 2] = b;
    pixels[i + 3] = 255;
  }
  image->UploadData(pixels.data());
  return image;
}

class OfflineBackendTest : public testing::Test {
 protected:
  void SetUp() override {
    const char *backend = std::getenv("SPARKIUM_TEST_BACKEND");
    const bool debug = std::getenv("SPARKIUM_TEST_DEBUG") != nullptr;
    ASSERT_EQ(graphics::CreateCore(backend ? ParseSparkiumBackend(backend) : graphics::BACKEND_API_DEFAULT,
                                   graphics::Core::Settings{2, debug}, &graphics),
              0);
    ASSERT_EQ(graphics->InitializeLogicalDeviceAutoSelect(false), 0);
    core = std::make_unique<sparkium::Core>(graphics.get());
  }

  // A unit quad facing the camera plus a camera and film, i.e. the smallest
  // scene every offline backend can render.
  struct QuadScene {
    std::unique_ptr<sparkium::Scene> scene;
    std::unique_ptr<sparkium::Camera> camera;
    std::unique_ptr<sparkium::Film> film;
    std::unique_ptr<sparkium::GeometryMesh> geometry;
    std::unique_ptr<sparkium::MaterialPrincipled> material;
    std::unique_ptr<sparkium::EntityGeometryMaterial> entity;
  };

  QuadScene MakeQuadScene(bool generate_tangents = false, sparkium::Material *material_override = nullptr) {
    QuadScene result;
    std::vector<Vector3<float>> positions{{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    std::vector<Vector2<float>> tex_coords{{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    uint32_t indices[]{0, 1, 2, 0, 2, 3};
    Mesh<> mesh(4, 6, indices, positions.data(), nullptr, tex_coords.data());
    if (generate_tangents)
      mesh.GenerateTangents();
    result.geometry = std::make_unique<sparkium::GeometryMesh>(core.get(), mesh);
    if (!material_override) {
      result.material = std::make_unique<sparkium::MaterialPrincipled>(core.get(), glm::vec3(0.8f));
      result.material->roughness = 0.4f;
    }
    result.entity = std::make_unique<sparkium::EntityGeometryMaterial>(
        core.get(), result.geometry.get(), material_override ? material_override : result.material.get());
    result.scene = std::make_unique<sparkium::Scene>(core.get());
    result.scene->AddEntity(result.entity.get());
    result.camera = std::make_unique<sparkium::Camera>(core.get(), glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0),
                                                                              glm::vec3(0, 1, 0)),
                                                       glm::radians(60.0f), 1.0f);
    result.film = std::make_unique<sparkium::Film>(core.get(), 32, 32);
    return result;
  }

  std::unique_ptr<graphics::Core> graphics;
  std::unique_ptr<sparkium::Core> core;
};

// The offline backends generate only the Sobol rows they consume; the rows must
// be bit-identical to the 65536 x 1024 table the online backends upload.
TEST_F(OfflineBackendTest, SobolRowsMatchTheOnlineTable) {
  const std::string direction_file = long_march::FindAssetFile("data/new-joe-kuo-7.21201");
  constexpr uint32_t kRows = 256;
  const std::vector<uint32_t> partial = sparkium::backends::GenerateSobolRows(kRows, direction_file);
  const std::vector<uint32_t> online =
      SobolTableGen(kRows, sparkium::backends::kSobolDimensions, direction_file);
  ASSERT_EQ(partial.size(), static_cast<size_t>(kRows) * sparkium::backends::kSobolDimensions);
  ASSERT_EQ(partial.size(), online.size());
  EXPECT_EQ(0, std::memcmp(partial.data(), online.data(), online.size() * sizeof(uint32_t)))
      << "the partial generator differs from grassland::SobolTableGen";

  // The full-size table starts with the same rows, so a lazily grown table is a
  // prefix of the one the graphics backends upload.
  const std::vector<uint32_t> full =
      SobolTableGen(1024, sparkium::backends::kSobolDimensions, direction_file);
  ASSERT_LE(partial.size(), full.size());
  EXPECT_EQ(0, std::memcmp(partial.data(), full.data(), partial.size() * sizeof(uint32_t)));
}

// Every registered image must be addressed separately: an offset in texels
// instead of floats used to make index 1 read index 0's pixels.
TEST_F(OfflineBackendTest, TextureTableAddressesEveryRegisteredImage) {
  QuadScene quad = MakeQuadScene();
  auto base_color = ConstantImage(graphics.get(), 255, 0, 0);
  auto normal = ConstantImage(graphics.get(), 0, 255, 0);
  quad.material->textures.base_color = base_color.get();
  quad.material->textures.normal = normal.get();
  auto offline = sparkium::backends::OfflineScene::Build(quad.scene.get(), quad.camera.get());

  const sparkium::backends::MaterialData &material = offline->Materials().at(0);
  const int32_t base_index =
      material.principled.texture_index[sparkium::backends::PRINCIPLED_TEXTURE_BASE_COLOR];
  const int32_t normal_index = material.principled.texture_index[sparkium::backends::PRINCIPLED_TEXTURE_NORMAL];
  ASSERT_GE(base_index, 0);
  ASSERT_GE(normal_index, 0);
  EXPECT_NE(base_index, normal_index);
  EXPECT_EQ(2u, offline->Textures().size());

  auto sample = [&](int32_t index) {
    const sparkium::backends::float4 value =
        sparkium::backends::SampleTexture(offline->Device(), index, sparkium::backends::float2(0.25f, 0.75f));
    return std::vector<float>{value.x, value.y, value.z};
  };
  EXPECT_EQ(sample(base_index), (std::vector<float>{1.0f, 0.0f, 0.0f}));
  EXPECT_EQ(sample(normal_index), (std::vector<float>{0.0f, 1.0f, 0.0f}));
  // Unregistered slots must fall back to the white default.
  EXPECT_EQ(sample(-1), (std::vector<float>{1.0f, 1.0f, 1.0f}));
}

// The flattening copies the graphics geometry buffer verbatim; the offline BVH
// and the HLSL hit records therefore read the same mesh layout.
TEST_F(OfflineBackendTest, FlattenedMeshMatchesTheGeometryBuffer) {
  QuadScene quad = MakeQuadScene(true);
  auto offline = sparkium::backends::OfflineScene::Build(quad.scene.get(), quad.camera.get());
  const std::vector<uint8_t> online = DownloadBuffer(quad.geometry->GetBuffer());
  const std::vector<uint8_t> &mesh_data = offline->MeshData();
  const sparkium::backends::MeshRange &range = offline->Meshes().at(0);
  ASSERT_LE(range.offset + online.size(), mesh_data.size());
  EXPECT_EQ(0, std::memcmp(mesh_data.data() + range.offset, online.data(), online.size()));
  // The device view points at the same flattened arrays.
  ASSERT_EQ(1u, offline->Device().num_meshes);
  EXPECT_EQ(range.offset, offline->Device().meshes[0].offset);
  EXPECT_EQ(range.root, offline->Device().meshes[0].root);
  EXPECT_EQ(mesh_data.data(), offline->Device().mesh_data);
  EXPECT_EQ(2u, range.primitive_count);
}

// The CPU backend must not need any graphics dispatch to shade: it runs on host
// workers, is deterministic, and produces a shaded (non-uniform) image.
TEST_F(OfflineBackendTest, CpuBackendShadesOnTheHost) {
  QuadScene quad = MakeQuadScene();
  quad.scene->settings.samples_per_dispatch = 8;
  quad.scene->settings.max_bounces = 2;
  quad.scene->settings.background_color = glm::vec3(0.5f, 0.6f, 0.7f);

  auto backend = sparkium::backends::CreateOfflineBackend("cpu");
  ASSERT_EQ(std::string("cpu"), backend->Name());
  const sparkium::backends::OfflineRenderResult first =
      sparkium::backends::RenderOfflineScene(backend.get(), quad.scene.get(), quad.camera.get(), quad.film.get(), 1);
  // The film keeps its accumulated sample counter between renders, exactly like
  // sparkium::Core::Render, so re-create it for a repeatable comparison.
  quad.film = std::make_unique<sparkium::Film>(core.get(), 32, 32);
  const sparkium::backends::OfflineRenderResult second =
      sparkium::backends::RenderOfflineScene(backend.get(), quad.scene.get(), quad.camera.get(), quad.film.get(), 1);
  EXPECT_EQ(first.rgba8, second.rgba8) << "the CPU backend is not deterministic";
  EXPECT_EQ(32u * 32u * 4, first.rgba8.size());
  // A shaded image must contain more than the plain background color: the quad
  // is visible against `background_color`.
  std::set<std::array<uint8_t, 3>> colors;
  for (size_t i = 0; i < first.rgba8.size(); i += 4)
    colors.insert({first.rgba8[i], first.rgba8[i + 1], first.rgba8[i + 2]});
  EXPECT_GE(colors.size(), 2u);
}

TEST_F(OfflineBackendTest, OfflineSceneSignatureTracksSceneState) {
  QuadScene quad = MakeQuadScene();
  const std::vector<uint64_t> first =
      sparkium::backends::OfflineSceneSignature(quad.scene.get(), quad.camera.get());
  EXPECT_EQ(first, sparkium::backends::OfflineSceneSignature(quad.scene.get(), quad.camera.get()));
  quad.scene->settings.samples_per_dispatch += 1;
  EXPECT_NE(first, sparkium::backends::OfflineSceneSignature(quad.scene.get(), quad.camera.get()));
}

// Materials the portable core cannot express must be rejected instead of being
// silently rendered with a placeholder.
TEST_F(OfflineBackendTest, ShaderGraphMaterialsAreRejected) {
  sparkium::MaterialShaderGraph shader_graph(core.get(), sparkium::CodeLines(std::string("return float4(1,0,0,1);")),
                                             {}, glm::vec3(0.0f));
  QuadScene quad = MakeQuadScene(false, &shader_graph);
  EXPECT_THROW(sparkium::backends::OfflineScene::Build(quad.scene.get(), quad.camera.get()), std::runtime_error);
}

}  // namespace

#if defined(LONGMARCH_CUDA_ENABLED)
// Both offline backends compile the same shading core; on a directly lit scene
// (a single bounce, no deep path recursion) their images must agree to within
// the difference between host and device floating point contraction.
TEST_F(OfflineBackendTest, CudaBackendMatchesCpuBackend) {
  QuadScene quad = MakeQuadScene();
  quad.scene->settings.samples_per_dispatch = 16;
  quad.scene->settings.max_bounces = 1;
  quad.scene->settings.background_color = glm::vec3(0.1f);
  sparkium::EntityPointLight light(core.get());
  light.position = glm::vec3(0.0f, 1.0f, 2.0f);
  light.color = glm::vec3(1.0f);
  light.strength = 40.0f;
  quad.scene->AddEntity(&light);

  auto cpu = sparkium::backends::CreateOfflineBackend("cpu");
  auto cuda = sparkium::backends::CreateOfflineBackend("cuda");
  ASSERT_EQ(std::string("cuda"), cuda->Name());
  EXPECT_FALSE(cuda->DeviceDescription().empty());
  const sparkium::backends::OfflineRenderResult cpu_result =
      sparkium::backends::RenderOfflineScene(cpu.get(), quad.scene.get(), quad.camera.get(), quad.film.get(), 1);
  const sparkium::backends::OfflineRenderResult cuda_result =
      sparkium::backends::RenderOfflineScene(cuda.get(), quad.scene.get(), quad.camera.get(), quad.film.get(), 1);

  ASSERT_EQ(cpu_result.rgba8.size(), cuda_result.rgba8.size());
  double difference = 0.0;
  double reference = 0.0;
  int max_difference = 0;
  for (size_t i = 0; i < cpu_result.rgba8.size(); ++i) {
    const int channel_difference =
        std::abs(static_cast<int>(cpu_result.rgba8[i]) - static_cast<int>(cuda_result.rgba8[i]));
    max_difference = std::max(max_difference, channel_difference);
    difference += channel_difference;
    reference += cpu_result.rgba8[i];
  }
  std::cout << "[ offline ] mean |cpu - cuda| = " << difference / cpu_result.rgba8.size()
            << ", max channel difference = " << max_difference << "\n";
  EXPECT_GT(reference, 0.0) << "the reference scene rendered black";
  EXPECT_LT(difference / cpu_result.rgba8.size(), 0.5)
      << "the CPU and CUDA backends disagree on a directly lit scene";
  // Both backends consume the same Sobol rows in the same order, so a pixel may
  // only move by the rounding of a float expression (a handful of levels); an
  // image-level divergence means one of them draws a different sample or takes
  // a different branch.
  EXPECT_LE(max_difference, 8) << "the CPU and CUDA backends diverge by more than rounding";
}
#else
TEST_F(OfflineBackendTest, CudaBackendIsReportedAsUnbuilt) {
  EXPECT_FALSE(sparkium::backends::OfflineBackendSupported("cuda"));
  EXPECT_THROW(sparkium::backends::CreateOfflineBackend("cuda"), std::runtime_error);
}
#endif
