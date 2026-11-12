#include "franka_scene.h"

#include <cmath>

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "glm/gtc/matrix_transform.hpp"
using namespace grassland;

namespace {
struct CombinedMesh {
  std::vector<std::unique_ptr<sparkium::GeometryMesh>> meshes;
  std::vector<std::unique_ptr<sparkium::MaterialPrincipled>> materials;
  std::vector<std::pair<std::unique_ptr<sparkium::EntityGeometryMaterial>, glm::mat4>> entities;

  void LoadEntities(sparkium::Core *core,
                    const aiScene *scene,
                    const aiNode *node,
                    const glm::mat4 transformation = {1.0f}) {
    glm::mat4 local_transform{
        node->mTransformation.a1, node->mTransformation.b1, node->mTransformation.c1, node->mTransformation.d1,
        node->mTransformation.a2, node->mTransformation.b2, node->mTransformation.c2, node->mTransformation.d2,
        node->mTransformation.a3, node->mTransformation.b3, node->mTransformation.c3, node->mTransformation.d3,
        node->mTransformation.a4, node->mTransformation.b4, node->mTransformation.c4, node->mTransformation.d4};
    if (scene->mRootNode == node) {
      local_transform = glm::mat4{1.0f};
    }
    local_transform = transformation * local_transform;
    for (int i = 0; i < node->mNumChildren; i++) {
      LoadEntities(core, scene, node->mChildren[i], local_transform);
    }
    for (int i = 0; i < node->mNumMeshes; i++) {
      int mesh_index = node->mMeshes[i];
      auto &mesh = meshes[mesh_index];
      auto &material = materials[scene->mMeshes[mesh_index]->mMaterialIndex];
      auto entity =
          std::make_unique<sparkium::EntityGeometryMaterial>(core, mesh.get(), material.get(), local_transform);
      entities.emplace_back(std::move(entity), local_transform);
    }
  }

  bool LoadModel(sparkium::Core *core, const std::string &path) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
      LogError("Failed to load model: " + path);
      return false;
    }

    for (int i = 0; i < scene->mNumMaterials; i++) {
      auto material = scene->mMaterials[i];
      aiColor4D color;
      material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
      float reflectivity = 0.0f;
      material->Get(AI_MATKEY_REFLECTIVITY, reflectivity);
      auto mat = std::make_unique<sparkium::MaterialPrincipled>(core, glm::vec3{color.r, color.g, color.b});
      mat->specular = reflectivity;
      mat->roughness = reflectivity;
      materials.emplace_back(std::move(mat));
    }

    for (int i = 0; i < scene->mNumMeshes; i++) {
      auto mesh = scene->mMeshes[i];
      std::vector<glm::vec3> positions;
      std::vector<glm::vec3> normals;
      std::vector<glm::vec2> tex_coords;
      std::vector<uint32_t> indices;

      for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
        positions.emplace_back(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
        if (mesh->HasNormals()) {
          normals.emplace_back(mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z);
        }
        if (mesh->HasTextureCoords(0)) {
          tex_coords.emplace_back(mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y);
        }
      }

      for (unsigned int f = 0; f < mesh->mNumFaces; f++) {
        auto face = mesh->mFaces[f];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
          indices.push_back(face.mIndices[j]);
        }
      }

      Mesh<> m(positions.size(), indices.size(), indices.data(), reinterpret_cast<Vector3<float> *>(positions.data()),
               mesh->HasNormals() ? reinterpret_cast<Vector3<float> *>(normals.data()) : nullptr,
               mesh->HasTextureCoords(0) ? reinterpret_cast<Vector2<float> *>(tex_coords.data()) : nullptr, nullptr);
      auto geom = std::make_unique<sparkium::GeometryMesh>(core, m);
      meshes.emplace_back(std::move(geom));
    }

    LoadEntities(core, scene, scene->mRootNode);

    // free scene and importer when out of scope
    importer.FreeScene();
    return core->IsValid();
  }

  void SetTransformation(const glm::mat4 &transform) {
    for (auto &[entity, local_transform] : entities) {
      entity->SetTransformation(transform * local_transform);
    }
  }

  void PutInScene(sparkium::Scene *scene) {
    for (auto &[entity, _] : entities) {
      scene->AddEntity(entity.get());
    }
  }
};

glm::mat4 xyz_rpy_trans(const glm::vec3 &xyz, const glm::vec3 &rpy) {
  glm::mat4 t = glm::translate(glm::mat4(1.0f), xyz);
  glm::mat4 r = glm::rotate(glm::mat4(1.0f), rpy.z, glm::vec3(0.0f, 0.0f, 1.0f)) *
                glm::rotate(glm::mat4(1.0f), rpy.y, glm::vec3(0.0f, 1.0f, 0.0f)) *
                glm::rotate(glm::mat4(1.0f), rpy.x, glm::vec3(1.0f, 0.0f, 0.0f));
  return t * r;
}

class AreaLight {
  const uint32_t indices[6] = {0, 2, 1, 0, 3, 2};
  const Vector3<float> vertices[4] = {{-1.0f, -1.0f, 0.0f},
                                      {1.0f, -1.0f, 0.0f},
                                      {1.0f, 1.0f, 0.0f},
                                      {-1.0f, 1.0f, 0.0f}};

 public:
  AreaLight(sparkium::Core *core,
            const glm::vec3 &emission = {1.0f, 1.0f, 1.0f},
            float size = 1.0f,
            const glm::vec3 &position = {0.0f, 0.0f, 0.0f},
            const glm::vec3 &direction = {0.0f, 0.0f, 1.0f},
            const glm::vec3 &up = {0.0f, 1.0f, 0.0f})
      : light_(core, emission, false, false),
        emission(light_.emission),
        position(position),
        size(size),
        direction(direction),
        up(up) {
    mesh_ = std::make_unique<sparkium::GeometryMesh>(core, Mesh<>{4, 6, indices, vertices});
    entity_geometry_material_ = std::make_unique<sparkium::EntityGeometryMaterial>(core, mesh_.get(), &light_);
    Sync();
  }

  void Sync() {
    entity_geometry_material_->SetTransformation(glm::inverse(glm::lookAt(position, position + direction, up)) *
                                                 glm::scale(glm::mat4{1.0f}, glm::vec3{size}));
  }

  operator sparkium::Entity *() {
    return entity_geometry_material_.get();
  }

  glm::vec3 &emission;
  float size{1.0f};
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  glm::vec3 direction{0.0f, -1.0f, 0.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};

  std::unique_ptr<sparkium::EntityGeometryMaterial> entity_geometry_material_;

 private:
  std::unique_ptr<sparkium::GeometryMesh> mesh_;
  sparkium::MaterialLight light_;
};

Mesh<> LoadCube() {
  Mesh<> mesh;
  mesh.LoadObjFile(FindAssetFile("meshes/cube.obj"));
  return mesh;
}
}  // namespace

struct FrankaScene::State {
  std::unique_ptr<sparkium::Core> owner;
  sparkium::Scene scene;
  CombinedMesh combined_mesh[11];
  sparkium::GeometryMesh sphere, cube;
  sparkium::MaterialPrincipled ground;
  sparkium::MaterialLight sky;
  sparkium::EntityGeometryMaterial ground_entity, sky_entity;
  AreaLight area_light;
  sparkium::Film film;
  sparkium::CameraPinhole camera;
  sparkium::Renderer renderer;
  bool ray_tracing{true};

  State(std::unique_ptr<sparkium::Core> core, int width, int height)
      : owner(std::move(core)),
        scene(owner.get()),
        sphere(owner.get(), Mesh<>::Sphere(30)),
        cube(owner.get(), LoadCube()),
        ground(owner.get(), {0.1f, 0.2f, 0.4f}),
        sky(owner.get(), {0.8f, 0.8f, 0.8f}, true, false),
        ground_entity(owner.get(),
                      &cube,
                      &ground,
                      glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, -1000.0f}) *
                          glm::scale(glm::mat4(1.0f), glm::vec3(1000.0f))),
        sky_entity(owner.get(),
                   &sphere,
                   &sky,
                   glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.0f}) *
                       glm::scale(glm::mat4(1.0f), glm::vec3(60.0f))),
        area_light(owner.get(),
                   glm::vec3{1.0f},
                   1.0f,
                   glm::vec3{40.0f, -30.0f, 30.0f},
                   glm::normalize(glm::vec3{-4.0f, 3.0f, -3.0f}),
                   glm::vec3{0.0f, 0.0f, 1.0f}),
        film(owner.get(), width, height),
        camera(owner.get(),
               glm::lookAt(glm::vec3{2.0f, -1.0f, 0.3f}, glm::vec3{0.0f, 0.0f, 0.5f}, glm::vec3{0.0, 0.0, 1.0}),
               glm::radians(30.0f),
               static_cast<float>(width) / height),
        renderer(owner.get()) {
    scene.settings.samples_per_dispatch = 32;
    scene.settings.raster.ambient_light = glm::vec3{0.5f};
    film.info.persistence = 0.98f;
    ground.roughness = 0.2f;
    ground.metallic = 0.0f;
    sky_entity.raster_light = false;
    area_light.emission = glm::vec3{1000.0f};
    scene.AddEntity(&ground_entity);
    scene.AddEntity(&sky_entity);
    scene.AddEntity(area_light);
  }

  bool Load() {
    std::string link_paths[] = {FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link0.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link1.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link2.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link3.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link4.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link5.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link6.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_arms/fr3/visual/link7.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_ee/franka_hand_white/visual/hand.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_ee/franka_hand_white/visual/finger.dae"),
                                FindAssetFile("urdfs/franka_fr3/meshes/robot_ee/franka_hand_white/visual/finger.dae")};
    for (int i = 0; i < 11; ++i) {
      if (!combined_mesh[i].LoadModel(owner.get(), link_paths[i]))
        return false;
      combined_mesh[i].PutInScene(&scene);
    }
    return film.IsValid() && renderer.IsValid();
  }

  bool Configure(const Settings &settings) {
    for (size_t i = 0; i < settings.joints.size(); ++i)
      if (!std::isfinite(settings.joints[i]) || settings.joints[i] < joint_lower[i] ||
          settings.joints[i] > joint_upper[i])
        return false;
    ray_tracing = settings.ray_tracing;
    glm::mat4 combined_mat = glm::mat4(1.0f);
    combined_mesh[1].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.333f}, {});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[0]});
    combined_mesh[1].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {-1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[1]});
    combined_mesh[2].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, -0.316f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[2]});
    combined_mesh[3].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0825f, 0.0f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[3]});
    combined_mesh[4].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({-0.0825f, 0.384f, 0.0f}, {-1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[4]});
    combined_mesh[5].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[5]});
    combined_mesh[6].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.088f, 0.0f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, settings.joints[6]});
    combined_mesh[7].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.107f}, {0.0f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -0.7853981633974483f});
    combined_mesh[8].SetTransformation(combined_mat);
    auto hand_link = combined_mat;
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.1034f}, {0.0f, 0.0f, 0.0f});
    combined_mat = hand_link * xyz_rpy_trans({0.0f, 0.0f, 0.0584f}, {0.0f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, settings.joints[7], 0.0f}, {0.0f, 0.0f, 0.0f});
    combined_mesh[9].SetTransformation(combined_mat);
    combined_mat = hand_link * xyz_rpy_trans({0.0f, 0.0f, 0.0584f}, {0.0f, 0.0f, 3.141592653589793f});
    combined_mat *= xyz_rpy_trans({0.0f, settings.joints[7], 0.0f}, {0.0f, 0.0f, 0.0f});
    combined_mesh[10].SetTransformation(combined_mat);

    // Joint and pipeline edits preserve the original demo's temporal history.
    return owner->IsValid();
  }
};

std::unique_ptr<FrankaScene> FrankaScene::Create(rhi::IDevice *device, int width, int height) {
  if (width <= 0 || height <= 0)
    return {};
  auto core = sparkium::Core::Create(device);
  if (!core)
    return {};
  auto state = std::make_unique<State>(std::move(core), width, height);
  if (!state->Load())
    return {};
  return std::unique_ptr<FrankaScene>(new FrankaScene(std::move(state)));
}

FrankaScene::FrankaScene(std::unique_ptr<State> state) : state_(std::move(state)) {
}

FrankaScene::~FrankaScene() = default;

bool FrankaScene::Configure(const Settings &settings) {
  return state_->Configure(settings);
}

sparkium::Film *FrankaScene::GetFilm() {
  return &state_->film;
}

rhi::ObjPtr<rhi::ISubmissionToken> FrankaScene::Submit(int samples,
                                                       rhi::IImage *output,
                                                       bool hdr,
                                                       const rhi::SubmitInfo &dependencies) {
  state_->scene.settings.samples_per_dispatch = samples;
  return state_->renderer.Submit(
      &state_->scene, &state_->camera, &state_->film,
      state_->ray_tracing ? sparkium::RENDER_PIPELINE_AUTO : sparkium::RENDER_PIPELINE_RASTERIZATION, output, hdr,
      dependencies);
}
