#include "snowberg/visualizer/visualizer_entity.h"

#include "snowberg/visualizer/visualizer_core.h"
#include "snowberg/visualizer/visualizer_film.h"
#include "snowberg/visualizer/visualizer_mesh.h"
#include "snowberg/visualizer/visualizer_ownership_holder.h"
#include "snowberg/visualizer/visualizer_render_context.h"

namespace snowberg::visualizer {

namespace {
#include "built_in_shaders.inl"
}

Entity::Entity(const std::shared_ptr<Core> &core) : core_(core) {
}

std::shared_ptr<Core> Entity::GetCore() const {
  return core_;
}

int Entity::ExecuteStage(RenderStage render_stage, const RenderContext &ctx) {
  return 0;
}

void Entity::PyBind(pybind11::module &m) {
  pybind11::class_<Entity, std::shared_ptr<Entity>> entity(m, "Entity");
  entity.def("get_core", &Entity::GetCore);

  EntityMeshObject::PyBind(m);
  EntityAmbientLight::PyBind(m);
  EntityDirectionalLight::PyBind(m);
}

EntityMeshObject::EntityMeshObject(const std::shared_ptr<Core> &core,
                                   const std::weak_ptr<Mesh> &mesh,
                                   const Material &material,
                                   const Matrix4<float> &transform)
    : Entity(core), mesh_(mesh) {
  program_ = core_->LoadProgram<ProgramWithGeometryShader>(PROGRAM_ID_NO_NORMAL, [&]() {
    std::shared_ptr<ProgramWithGeometryShader> program = std::make_shared<ProgramWithGeometryShader>();
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/entity.hlsl"), "VSMain", "vs_6_0",
                                        &program->vertex_shader_);
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/entity.hlsl"), "GSMain", "gs_6_0",
                                        &program->geometry_shader_);
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/entity.hlsl"), "PSMain", "ps_6_0",
                                        &program->fragment_shader_);
    core_->GraphicsCore()->CreateProgram(
        {FilmChannelImageFormat(FILM_CHANNEL_EXPOSURE), FilmChannelImageFormat(FILM_CHANNEL_ALBEDO),
         FilmChannelImageFormat(FILM_CHANNEL_POSITION), FilmChannelImageFormat(FILM_CHANNEL_NORMAL)},
        FilmChannelImageFormat(FILM_CHANNEL_DEPTH), &program->program_);
    program->program_->BindShader(program->vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program->program_->BindShader(program->geometry_shader_.get(), graphics::SHADER_TYPE_GEOMETRY);
    program->program_->BindShader(program->fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program->program_->AddInputBinding(sizeof(Vertex), false);
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, offsetof(Vertex, position));
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, offsetof(Vertex, normal));
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT2, offsetof(Vertex, tex_coord));
    program->program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT4, offsetof(Vertex, color));
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER,
                                          1);  // Camera Info
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER,
                                          1);  // Entity Info
    program->program_->SetCullMode(graphics::CULL_MODE_NONE);
    program->program_->Finalize();
    return program;
  });
  info_.material = material;
  info_.model = EigenToGLM(transform);
  core_->GraphicsCore()->CreateBuffer(sizeof(EntityInfo), graphics::BUFFER_TYPE_DYNAMIC, &info_buffer_);
}

int EntityMeshObject::ExecuteStage(RenderStage render_stage, const RenderContext &ctx) {
  std::shared_ptr<Mesh> mesh = mesh_.lock();
  if (mesh) {
    if (render_stage == RENDER_STAGE_RASTER_GEOMETRY_PASS) {
      ctx.ownership_holder->AddMesh(mesh);
      ctx.cmd_ctx->CmdBindProgram(program_->program_.get());
      ctx.cmd_ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
      ctx.cmd_ctx->CmdBindResources(0, {ctx.camera_buffer});
      info_buffer_->UploadData(&info_, sizeof(info_));
      ctx.cmd_ctx->CmdBindResources(1, {info_buffer_.get()});
      ctx.cmd_ctx->CmdBindVertexBuffers(0, {mesh->GetVertexBuffer()}, {0});
      ctx.cmd_ctx->CmdBindIndexBuffer(mesh->GetIndexBuffer(), 0);
      ctx.cmd_ctx->CmdDrawIndexed(mesh->IndexCount(), 1, 0, 0, 0);
    }
  } else {
    return -1;
  }
  return 0;
}

void EntityMeshObject::SetMesh(const std::shared_ptr<Mesh> &mesh) {
  mesh_ = mesh;
}

void EntityMeshObject::SetMaterial(const Material &material) {
  info_.material = material;
}

void EntityMeshObject::SetTransform(const Matrix4<float> &transform) {
  info_.model = EigenToGLM(transform);
}

void EntityMeshObject::PyBind(pybind11::module &m) {
  pybind11::class_<EntityMeshObject, Entity, std::shared_ptr<EntityMeshObject>> entity_mesh_object(m,
                                                                                                   "EntityMeshObject");
  entity_mesh_object.def("set_mesh", &EntityMeshObject::SetMesh, pybind11::arg("mesh") = nullptr);
  entity_mesh_object.def("set_material", &EntityMeshObject::SetMaterial, pybind11::arg("material") = Material{});
  entity_mesh_object.def("set_transform", &EntityMeshObject::SetTransform,
                         pybind11::arg("transform") = Matrix4<float>::Identity());
}

EntityAmbientLight::EntityAmbientLight(const std::shared_ptr<Core> &core, const Vector3<float> &intensity)
    : Entity(core) {
  program_ = core_->LoadProgram<ProgramCommonRaster>(PROGRAM_AMBIENT_LIGHTING_PASS, [&]() {
    std::shared_ptr<ProgramCommonRaster> program = std::make_shared<ProgramCommonRaster>();
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/ambient_light.hlsl"), "VSMain", "vs_6_0",
                                        &program->vertex_shader_);
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/ambient_light.hlsl"), "PSMain", "ps_6_0",
                                        &program->fragment_shader_);
    core_->GraphicsCore()->CreateProgram({FilmChannelImageFormat(FILM_CHANNEL_EXPOSURE)},
                                         graphics::IMAGE_FORMAT_UNDEFINED, &program->program_);
    program->program_->BindShader(program->vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program->program_->BindShader(program->fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Albedo
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Position
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Normal
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Depth
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER,
                                          1);  // Camera Info
    program->program_->SetCullMode(graphics::CULL_MODE_NONE);
    program->program_->SetBlendState(
        0, graphics::BlendState{graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD,
                                graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD});
    program->program_->Finalize();
    return program;
  });
  intensity_ = intensity;
  core_->GraphicsCore()->CreateBuffer(sizeof(Vector3<float>), graphics::BUFFER_TYPE_DYNAMIC, &intensity_buffer_);
}

int EntityAmbientLight::ExecuteStage(RenderStage render_stage, const RenderContext &ctx) {
  if (render_stage == RENDER_STAGE_RASTER_LIGHTING_PASS) {
    intensity_buffer_->UploadData(&intensity_, sizeof(intensity_));
    ctx.cmd_ctx->CmdBindProgram(program_->program_.get());
    ctx.cmd_ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    ctx.cmd_ctx->CmdBindResources(0, {ctx.film->GetImage(FILM_CHANNEL_ALBEDO)});
    ctx.cmd_ctx->CmdBindResources(1, {ctx.film->GetImage(FILM_CHANNEL_POSITION)});
    ctx.cmd_ctx->CmdBindResources(2, {ctx.film->GetImage(FILM_CHANNEL_NORMAL)});
    ctx.cmd_ctx->CmdBindResources(3, {ctx.film->GetImage(FILM_CHANNEL_DEPTH)});
    ctx.cmd_ctx->CmdBindResources(4, {intensity_buffer_.get()});

    ctx.cmd_ctx->CmdDraw(6, 1, 0, 0);
  }
  return 0;
}

void EntityAmbientLight::SetIntensity(const Vector3<float> &intensity) {
  intensity_ = intensity;
}

void EntityAmbientLight::PyBind(pybind11::module &m) {
  pybind11::class_<EntityAmbientLight, Entity, std::shared_ptr<EntityAmbientLight>> entity_ambient_light(
      m, "EntityAmbientLight");
  entity_ambient_light.def("set_intensity", &EntityAmbientLight::SetIntensity,
                           pybind11::arg("intensity") = Vector3<float>{0.8f, 0.8f, 0.8f});
}

EntityDirectionalLight::EntityDirectionalLight(const std::shared_ptr<Core> &core,
                                               const Vector3<float> &direction,
                                               const Vector3<float> &intensity)
    : Entity(core) {
  SetDirection(direction);
  SetIntensity(intensity);
  program_ = core_->LoadProgram<ProgramCommonRaster>(PROGRAM_DIRECTION_LIGHTING_PASS, [&]() {
    std::shared_ptr<ProgramCommonRaster> program = std::make_shared<ProgramCommonRaster>();
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/directional_light.hlsl"), "VSMain", "vs_6_0",
                                        &program->vertex_shader_);
    core_->GraphicsCore()->CreateShader(GetShaderCode("shaders/directional_light.hlsl"), "PSMain", "ps_6_0",
                                        &program->fragment_shader_);
    core_->GraphicsCore()->CreateProgram({FilmChannelImageFormat(FILM_CHANNEL_EXPOSURE)},
                                         graphics::IMAGE_FORMAT_UNDEFINED, &program->program_);
    program->program_->BindShader(program->vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program->program_->BindShader(program->fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Albedo
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Position
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Normal
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE,
                                          1);  // Depth
    program->program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER,
                                          1);  // Camera Info
    program->program_->SetCullMode(graphics::CULL_MODE_NONE);
    program->program_->SetBlendState(
        0, graphics::BlendState{graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD,
                                graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD});
    program->program_->Finalize();
    return program;
  });

  core_->GraphicsCore()->CreateBuffer(sizeof(LightInfo), graphics::BUFFER_TYPE_DYNAMIC, &light_info_buffer_);
}

int EntityDirectionalLight::ExecuteStage(RenderStage render_stage, const RenderContext &ctx) {
  if (render_stage == RENDER_STAGE_RASTER_LIGHTING_PASS) {
    LightInfo info;
    info.direction = direction_;
    info.intensity = intensity_;
    light_info_buffer_->UploadData(&info, sizeof(info));
    ctx.cmd_ctx->CmdBindProgram(program_->program_.get());
    ctx.cmd_ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    ctx.cmd_ctx->CmdBindResources(0, {ctx.film->GetImage(FILM_CHANNEL_ALBEDO)});
    ctx.cmd_ctx->CmdBindResources(1, {ctx.film->GetImage(FILM_CHANNEL_POSITION)});
    ctx.cmd_ctx->CmdBindResources(2, {ctx.film->GetImage(FILM_CHANNEL_NORMAL)});
    ctx.cmd_ctx->CmdBindResources(3, {ctx.film->GetImage(FILM_CHANNEL_DEPTH)});
    ctx.cmd_ctx->CmdBindResources(4, {light_info_buffer_.get()});

    ctx.cmd_ctx->CmdDraw(6, 1, 0, 0);
  }
  return 0;
}

void EntityDirectionalLight::SetIntensity(const Vector3<float> &intensity) {
  intensity_ = intensity;
}

void EntityDirectionalLight::SetDirection(const Vector3<float> &direction) {
  direction_ = direction.normalized();
}

void EntityDirectionalLight::PyBind(pybind11::module &m) {
  pybind11::class_<EntityDirectionalLight, Entity, std::shared_ptr<EntityDirectionalLight>> entity_directional_light(
      m, "EntityDirectionalLight");
  entity_directional_light.def("set_intensity", &EntityDirectionalLight::SetIntensity,
                               pybind11::arg("intensity") = Vector3<float>{0.8f, 0.8f, 0.8f});
  entity_directional_light.def("set_direction", &EntityDirectionalLight::SetDirection,
                               pybind11::arg("direction") = Vector3<float>{3.0f, 1.0f, 2.0f});
}

}  // namespace snowberg::visualizer
