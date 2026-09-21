// This executable links only the scene model and loader, not graphics or a renderer.
#include <gtest/gtest.h>

#include "scene_fixture.h"
#include "sparkium/scene_io/json_scene.h"

using namespace sparkium;

TEST(SceneDefinition, LoadsCompleteHostDataWithoutDeviceAndSurvivesDeletedSources) {
  sparkium_test::SceneFiles files;
  auto scene = LoadScene(files.directory / "scene.json");
  files.RemoveSources();
  EXPECT_EQ(scene->name, "Memory scene");
  const auto &mesh = std::get<grassland::Mesh<float>>(scene->geometries.at("triangle")->shape);
  EXPECT_EQ(mesh.NumVertices(), 3);
  EXPECT_EQ(mesh.NumIndices(), 3);
  auto texture = scene->materials.at("surface").textures.at("base_color");
  EXPECT_EQ(texture->rgba, (std::vector<uint8_t>{255, 128, 64, 255}));
  EXPECT_EQ(texture, scene->materials.at("surface").textures.at("emission"));
  EXPECT_EQ(texture, scene->materials.at("graph").graph_textures.at("image"));
  auto &graph = std::get<NodeValue::Object>(scene->materials.at("graph").graph.value);
  auto &nodes = std::get<NodeValue::Object>(graph.at("nodes").value);
  EXPECT_EQ(std::get<NodeValue::Object>(nodes.at("image").value).count("path"), 0);
  EXPECT_NO_THROW(scene->Validate());
}

TEST(SceneDefinition, ProgrammaticSnapshotsShareAssetsButNotMutableSettings) {
  sparkium_test::SceneFiles files;
  auto original = LoadScene(files.directory / "scene.json");
  SceneDefinition edited = *original;
  edited.integrator.samples_per_dispatch = 7;
  edited.materials.at("surface").principled.roughness = 0.9f;
  EXPECT_EQ(original->integrator.samples_per_dispatch, 2);
  EXPECT_EQ(original->materials.at("surface").principled.roughness, 0.5f);
  EXPECT_EQ(original->geometries.at("triangle"), edited.geometries.at("triangle"));
  EXPECT_NO_THROW(edited.Validate());
}

TEST(SceneDefinition, InvalidReferencesAndIncompleteTexturesAreRejected) {
  SceneDefinition scene;
  scene.entities.emplace_back(InstanceDefinition{"missing", "missing"});
  EXPECT_THROW(scene.Validate(), std::invalid_argument);
  scene.entities.clear();
  auto texture = std::make_shared<TextureData>();
  texture->width = texture->height = 2;
  scene.materials["material"].textures["base_color"] = texture;
  EXPECT_THROW(scene.Validate(), std::invalid_argument);
}

TEST(SceneDefinition, MissingFilesFailDuringLoad) {
  sparkium_test::SceneFiles files;
  std::filesystem::remove(files.directory / "texture.tga");
  EXPECT_THROW(LoadScene(files.directory / "scene.json"), std::runtime_error);
}
