#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <random>

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "glm/gtc/matrix_transform.hpp"
#include "stb_image_write.h"

using namespace long_march;

struct CombinedMesh {
  std::vector<std::unique_ptr<sparkium::GeometryMesh>> meshes;
  std::vector<std::unique_ptr<sparkium::MaterialPrincipled>> materials;
  std::vector<std::pair<std::unique_ptr<sparkium::EntityGeometryMaterial>, glm::mat4>> entities;

  void LoadEntities(sparkium::Core *core,
                    const aiScene *scene,
                    const aiNode *node,
                    const glm::mat4 transformation = {1.0f}) {
    // Show transformation
    glm::mat4 local_transform{
        node->mTransformation.a1, node->mTransformation.b1, node->mTransformation.c1, node->mTransformation.d1,
        node->mTransformation.a2, node->mTransformation.b2, node->mTransformation.c2, node->mTransformation.d2,
        node->mTransformation.a3, node->mTransformation.b3, node->mTransformation.c3, node->mTransformation.d3,
        node->mTransformation.a4, node->mTransformation.b4, node->mTransformation.c4, node->mTransformation.d4};
    if (scene->mRootNode == node) {
      local_transform = glm::mat4{1.0f};
    }
    local_transform = transformation * local_transform;
    // LogInfo("{} {} {} {}", local_transform[0][0], local_transform[1][0], local_transform[2][0],
    // local_transform[3][0]); LogInfo("{} {} {} {}", local_transform[0][1], local_transform[1][1],
    // local_transform[2][1], local_transform[3][1]); LogInfo("{} {} {} {}", local_transform[0][2],
    // local_transform[1][2], local_transform[2][2], local_transform[3][2]); LogInfo("{} {} {} {}",
    // local_transform[0][3], local_transform[1][3], local_transform[2][3], local_transform[3][3]);
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

  void LoadModel(sparkium::Core *core, const std::string &path) {
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
      LogError("Failed to load model: " + path);
      return;
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

  void Clear() {
    entities.clear();
    meshes.clear();
    materials.clear();
  }

} combined_mesh[11];

glm::mat4 xyz_rpy_trans(const glm::vec3 &xyz, const glm::vec3 &rpy) {
  glm::mat4 t = glm::translate(glm::mat4(1.0f), xyz);
  glm::mat4 r = glm::rotate(glm::mat4(1.0f), rpy.z, glm::vec3(0.0f, 0.0f, 1.0f)) *
                glm::rotate(glm::mat4(1.0f), rpy.y, glm::vec3(0.0f, 1.0f, 0.0f)) *
                glm::rotate(glm::mat4(1.0f), rpy.x, glm::vec3(1.0f, 0.0f, 0.0f));
  return t * r;
}

struct JointInfo {
  float lower_bound{0.0f};
  float upper_bound{0.0f};
  float value{0.0f};
};

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

int main() {
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

  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{2, false}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparkium::Core sparkium_core(core_.get());
  sparkium_core.GetShadersVFS().Print();

  sparkium::Scene scene(&sparkium_core);
  scene.settings.samples_per_dispatch = 32;
  sparkium::Film film(&sparkium_core, 1024, 512);
  film.info.persistence = 0.98f;
  sparkium::Camera camera(
      &sparkium_core, glm::lookAt(glm::vec3{2.0f, -1.0f, 0.3f}, glm::vec3{0.0f, 0.0f, 0.5f}, glm::vec3{0.0, 0.0, 1.0}),
      glm::radians(30.0f), static_cast<float>(film.GetWidth()) / film.GetHeight());

  Mesh<> matball_mesh;
  matball_mesh.LoadObjFile(FindAssetFile("meshes/preview_sphere.obj"));
  matball_mesh.GenerateTangents();
  Mesh<> cube_mesh;
  cube_mesh.LoadObjFile(FindAssetFile("meshes/cube.obj"));
  sparkium::GeometryMesh geometry_sphere(&sparkium_core, Mesh<>::Sphere(30));
  sparkium::GeometryMesh geometry_matball(&sparkium_core, matball_mesh);
  sparkium::GeometryMesh geometry_cube(&sparkium_core, cube_mesh);

  sparkium::MaterialPrincipled material_ground(&sparkium_core, {0.1f, 0.2f, 0.4f});
  material_ground.roughness = 0.2f;
  material_ground.metallic = 0.0f;
  sparkium::EntityGeometryMaterial entity_ground(&sparkium_core, &geometry_cube, &material_ground,
                                                 glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, -1000.0f}) *
                                                     glm::scale(glm::mat4(1.0f), glm::vec3(1000.0f)));

  sparkium::MaterialLight material_sky(&sparkium_core, {0.8f, 0.8f, 0.8f}, true, false);
  sparkium::EntityGeometryMaterial entity_sky(
      &sparkium_core, &geometry_sphere, &material_sky,
      glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 0.0f, 0.0f}) * glm::scale(glm::mat4(1.0f), glm::vec3(60.0f)));
  entity_sky.raster_light = false;
  scene.settings.raster.ambient_light = glm::vec3{0.8f, 0.8f, 0.8f};
  AreaLight area_light(&sparkium_core, glm::vec3{1.0f, 1.0f, 1.0f}, 1.0f, glm::vec3{40.0f, -30.0f, 30.0f},
                       glm::normalize(glm::vec3{-4.0f, 3.0f, -3.0f}), glm::vec3{0.0f, 0.0f, 1.0f});
  area_light.emission = glm::vec3{1000.0f};

  scene.AddEntity(&entity_ground);
  scene.AddEntity(&entity_sky);
  scene.AddEntity(area_light);

  for (int i = 0; i < 11; i++) {
    combined_mesh[i].LoadModel(&sparkium_core, link_paths[i]);
    combined_mesh[i].PutInScene(&scene);
  }

  std::unique_ptr<graphics::Image> srgb_image;
  core_->CreateImage(film.GetWidth(), film.GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &srgb_image);

  std::unique_ptr<graphics::Window> window;
  core_->CreateWindowObject(film.GetWidth(), film.GetHeight(), "Sparkium", &window);
  FPSCounter fps_counter;

  JointInfo joints[] = {{-2.7437f, 2.7437f, 0.0f},      {-1.7837f, 1.7837f, 0.0f}, {-2.9007f, 2.9007f, 0.0f},
                        {-3.0421f, -0.1518f, -0.1518f}, {-2.8065f, 2.8065f, 0.0f}, {0.5445f, 4.5169f, 0.5445f},
                        {-3.0159f, 3.01599f, 0.0f},     {0.0f, 0.04f, 0.0f}};

  window->InitImGui(nullptr, 26.0f);
  while (!window->ShouldClose()) {
    window->BeginImGuiFrame();
    if (ImGui::Begin("Franka Joint Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      if (ImGui::Button("Center")) {
        for (auto &j : joints) {
          j.value = (j.upper_bound + j.lower_bound) * 0.5f;
        }
      }
      for (int i = 0; i < 8; i++) {
        ImGui::SliderFloat(("Joint " + std::to_string(i + 1)).c_str(), &joints[i].value, joints[i].lower_bound,
                           joints[i].upper_bound);
      }
    }
    ImGui::End();
    window->EndImGuiFrame();
    glm::mat4 combined_mat = glm::mat4(1.0f);
    combined_mesh[1].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.333f}, {});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[0].value});
    combined_mesh[1].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {-1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[1].value});
    combined_mesh[2].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, -0.316f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[2].value});
    combined_mesh[3].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0825f, 0.0f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[3].value});
    combined_mesh[4].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({-0.0825f, 0.384f, 0.0f}, {-1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[4].value});
    combined_mesh[5].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[5].value});
    combined_mesh[6].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.088f, 0.0f, 0.0f}, {1.570796326794897f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, joints[6].value});
    combined_mesh[7].SetTransformation(combined_mat);
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.107f}, {0.0f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -0.7853981633974483f});
    combined_mesh[8].SetTransformation(combined_mat);
    auto hand_link = combined_mat;
    combined_mat *= xyz_rpy_trans({0.0f, 0.0f, 0.1034f}, {0.0f, 0.0f, 0.0f});
    auto hand_tcp_link = combined_mat;
    combined_mat = hand_link * xyz_rpy_trans({0.0f, 0.0f, 0.0584f}, {0.0f, 0.0f, 0.0f});
    combined_mat *= xyz_rpy_trans({0.0f, joints[7].value, 0.0f}, {0.0f, 0.0f, 0.0f});
    combined_mesh[9].SetTransformation(combined_mat);
    combined_mat = hand_link * xyz_rpy_trans({0.0f, 0.0f, 0.0584f}, {0.0f, 0.0f, 3.141592653589793f});
    combined_mat *= xyz_rpy_trans({0.0f, joints[7].value, 0.0f}, {0.0f, 0.0f, 0.0f});
    combined_mesh[10].SetTransformation(combined_mat);

    // area_light.position = glm::mat3{glm::rotate(glm::mat4{1.0f}, glm::radians(0.3f), glm::vec3{0.0f, 1.0f,
    // 0.0f})} * area_light.position; if (area_light.position.y < 0.0) area_light.position = -area_light.position;
    // area_light.direction = -area_light.position;
    sparkium_core.Render(&scene, &camera, &film);
    film.Develop(srgb_image.get());
    std::unique_ptr<graphics::CommandContext> cmd_context;
    core_->CreateCommandContext(&cmd_context);
    cmd_context->CmdPresent(window.get(), srgb_image.get());
    core_->SubmitCommandContext(cmd_context.get());
    glfwPollEvents();
    float fps = fps_counter.TickFPS();
    char fps_buf[16];
    sprintf(fps_buf, "%.2f", fps);
    float rps = film.GetWidth() * film.GetHeight() * fps * scene.settings.samples_per_dispatch;
    char rps_buf[16];
    sprintf(rps_buf, "%.2f", rps * 1e-6f);
    window->SetTitle(std::string("Franka Kinematics - ") + fps_buf + "frames/s" + " - " + rps_buf + "Mrays/s");
  }

  for (auto &cm : combined_mesh) {
    cm.Clear();
  }

  film.Develop(srgb_image.get());
  std::vector<uint8_t> image_data(film.GetWidth() * film.GetHeight() * 4);
  srgb_image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", film.GetWidth(), film.GetHeight(), 4, image_data.data());
}
