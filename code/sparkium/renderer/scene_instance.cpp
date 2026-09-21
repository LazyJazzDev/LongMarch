#include "sparkium/renderer/scene_instance.h"

#include "sparkium/renderer/shader_graph.h"
#ifdef SPARKIUM_NATIVE_ENABLED
#include "sparkium/backend/common/native_image.h"
#endif

namespace sparkium::detail {
SceneInstance::SceneInstance(Core *core, std::shared_ptr<const SceneDefinition> source)
    : definition(std::move(source)) {
  if (!definition)
    throw std::invalid_argument("null scene definition");
  definition->Validate();
  auto *device = core->BackendDevice();
  auto texture_image = [&](const std::shared_ptr<const TextureData> &texture) -> graphics::Image * {
    if (auto it = images.find(texture.get()); it != images.end())
      return it->second.get();
    std::unique_ptr<graphics::Image> image;
#ifdef SPARKIUM_NATIVE_ENABLED
    if (device->API() == RenderBackend::CPU) {
      image = std::make_unique<backend::NativeImage>(texture);
    } else
#endif
    {
      if (device->CreateImage(texture->width, texture->height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image))
        throw std::runtime_error("failed to create scene texture");
      image->UploadData(texture->rgba.data());
    }
    auto *result = image.get();
    images.emplace(texture.get(), std::move(image));
    return result;
  };
  for (const auto &[id, spec] : definition->materials) {
    std::unique_ptr<Material> material;
    switch (spec.type) {
      case MaterialDefinition::Type::Lambertian:
        material = std::make_unique<MaterialLambertian>(core, spec.base_color, spec.emission);
        break;
      case MaterialDefinition::Type::Specular:
        material = std::make_unique<MaterialSpecular>(core, spec.base_color);
        break;
      case MaterialDefinition::Type::Light:
        material = std::make_unique<MaterialLight>(core, spec.emission, spec.two_sided, spec.block_ray,
                                                   spec.camera_visible, spec.falloff_distance);
        break;
      case MaterialDefinition::Type::Principled: {
        auto principled = std::make_unique<MaterialPrincipled>(core, spec.principled.base_color);
        principled->info = spec.principled;
        const std::pair<const char *, graphics::Image **> slots[] = {
            {"normal", &principled->textures.normal},
            {"base_color", &principled->textures.base_color},
            {"metallic", &principled->textures.metallic},
            {"specular", &principled->textures.specular},
            {"roughness", &principled->textures.roughness},
            {"anisotropic", &principled->textures.anisotropic},
            {"anisotropic_rotation", &principled->textures.anisotropic_rotation},
            {"emission", &principled->textures.emission}};
        for (auto [slot, destination] : slots)
          if (auto it = spec.textures.find(slot); it != spec.textures.end())
            *destination = texture_image(it->second);
        principled->textures.normal_reverse_y = spec.normal_reverse_y;
        material = std::move(principled);
        break;
      }
      case MaterialDefinition::Type::ShaderGraph: {
        std::map<std::string, int> slots;
        std::vector<graphics::Image *> textures;
        for (const auto &[node, texture] : spec.graph_textures) {
          slots[node] = static_cast<int>(textures.size());
          textures.push_back(texture_image(texture));
        }
        material = std::make_unique<MaterialShaderGraph>(core, CompileShaderGraph(spec.graph, slots), textures,
                                                         spec.emission_hint);
        break;
      }
    }
    materials.emplace(id, std::move(material));
  }
  for (const auto &[id, geometry] : definition->geometries) {
    if (auto mesh = std::get_if<grassland::Mesh<float>>(&geometry->shape)) {
      // Aliasing ownership keeps the original mesh alive without copying it.
      auto shared_mesh = std::shared_ptr<const grassland::Mesh<float>>(geometry, mesh);
      geometries.emplace(id, std::make_unique<GeometryMesh>(core, std::move(shared_mesh)));
    } else {
      const auto &hair = std::get<HairData>(geometry->shape);
      geometries.emplace(
          id, std::make_unique<GeometryHair>(core, hair.points, hair.radii, hair.offsets, hair.radial_segments));
    }
  }
  scene = std::make_unique<Scene>(core);
  const auto &settings = definition->integrator;
  scene->settings.samples_per_dispatch = settings.samples_per_dispatch;
  scene->settings.max_bounces = settings.max_bounces;
  scene->settings.alpha_shadow = settings.alpha_shadow;
  scene->settings.background_color = settings.background_color;
  scene->settings.ambient_light = settings.ambient_light;
  for (const auto &spec : definition->entities) {
    std::unique_ptr<Entity> entity;
    bool active;
    if (auto instance = std::get_if<InstanceDefinition>(&spec)) {
      auto mesh = std::make_unique<EntityGeometryMaterial>(core, geometries.at(instance->geometry).get(),
                                                           materials.at(instance->material).get(), instance->transform);
      mesh->raster_light = instance->raster_light;
      entity = std::move(mesh);
      active = instance->active;
    } else {
      const auto &light = std::get<PointLightDefinition>(spec);
      entity = std::make_unique<EntityPointLight>(core, light.position, light.color, light.strength, light.radius,
                                                  light.soft_falloff, light.sampling_weight);
      active = light.active;
    }
    scene->AddEntity(entity.get());
    scene->SetEntityActive(entity.get(), active);
    entities.push_back(std::move(entity));
  }
  const auto &c = definition->camera;
  camera = std::make_unique<Camera>(core, c.view, c.fovy, c.aspect);
  camera->aperture_radius = c.aperture_radius;
  camera->focus_distance = c.focus_distance;
  camera->aperture_blades = c.aperture_blades;
  camera->aperture_rotation = c.aperture_rotation;
  camera->aperture_ratio = c.aperture_ratio;
  const auto &f = definition->film;
  film = std::make_unique<Film>(core, f.width, f.height);
  film->info.persistence = f.persistence;
  film->info.clamping = f.clamping;
  film->info.max_exposure = f.max_exposure;
  film->info.view_transform = f.view_transform;
  film->info.exposure = f.exposure;
  film->info.gamma = f.gamma;
  film->info.contrast = f.contrast;
  film->Reset();
}
}  // namespace sparkium::detail
