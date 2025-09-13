#include "app.h"

#include <glm/gtc/matrix_transform.hpp>
namespace {
#include "built_in_shaders.inl"
}

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  grassland::LogInfo("Device Name: {}", core_->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(1280, 720,
                            ((core_->API() == grassland::graphics::BACKEND_API_VULKAN) ? "[Vulkan]" : "[D3D12]") +
                                std::string(" Graphics Hello Resize"),
                            false, true, &window_);

  std::vector<Vertex> vertices = {
      {{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},  {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
      {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},   {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f}},
      {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}}, {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 1.0f}},
      {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 1.0f}},  {{1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}}};

  std::vector<uint32_t> indices = {0, 1, 2, 2, 1, 3, 2, 3, 6, 6, 3, 7, 6, 7, 4, 4, 7, 5,
                                   4, 5, 0, 0, 5, 1, 1, 5, 3, 3, 5, 7, 0, 2, 4, 4, 2, 6};

  core_->CreateBuffer(vertices.size() * sizeof(Vertex), grassland::graphics::BUFFER_TYPE_STATIC, &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_STATIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  core_->CreateBuffer(sizeof(GlobalUniformBuffer), grassland::graphics::BUFFER_TYPE_DYNAMIC, &uniform_buffer_);

  core_->CreateImage(1280, 720, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);
  core_->CreateImage(1280, 720, grassland::graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_image_);

  window_->ResizeEvent().RegisterCallback([this](int width, int height) {
    core_->WaitGPU();
    color_image_.reset();
    depth_image_.reset();
    core_->CreateImage(width, height, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);
    core_->CreateImage(width, height, grassland::graphics::IMAGE_FORMAT_D32_SFLOAT, &depth_image_);
  });

  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "PSMain", "ps_6_0", &fragment_shader_);
  grassland::LogInfo("Shader compiled successfully");

  core_->CreateProgram({grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT},
                       grassland::graphics::IMAGE_FORMAT_D32_SFLOAT, &program_);
  program_->AddInputBinding(sizeof(Vertex), false);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, sizeof(float) * 3);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->SetCullMode(grassland::graphics::CULL_MODE_NONE);
  program_->BindShader(vertex_shader_.get(), grassland::graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), grassland::graphics::SHADER_TYPE_PIXEL);
  program_->Finalize();
}

void Application::OnClose() {
  program_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  color_image_.reset();
  depth_image_.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
  uniform_buffer_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
  if (alive_) {
    static float x = 0.0;
    x += glm::radians(1.0f);
    while (x > glm::radians(360.0f)) {
      x -= glm::radians(360.0f);
    }

    auto extent = color_image_->Extent();
    GlobalUniformBuffer ubo = {};
    ubo.model = glm::rotate(glm::mat4{1.0f}, x, glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.view = glm::lookAt(glm::vec3{0.0f, 0.0f, 5.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    ubo.proj = glm::perspectiveZO(glm::radians(45.0f), float(extent.width) / float(extent.height), 3.5f, 6.5f);
    uniform_buffer_->UploadData(&ubo, sizeof(GlobalUniformBuffer));
  }
}

void Application::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdClearImage(depth_image_.get(), {{1.0f}});
  command_context->CmdBeginRendering({color_image_.get()}, depth_image_.get());
  command_context->CmdBindProgram(program_.get());
  command_context->CmdBindVertexBuffers(0, {vertex_buffer_.get()}, {0});
  command_context->CmdBindIndexBuffer(index_buffer_.get(), 0);
  command_context->CmdBindResources(0, {uniform_buffer_.get()});

  auto extent = color_image_->Extent();
  command_context->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0.0f, 1.0f});
  command_context->CmdSetScissor({0, 0, extent.width, extent.height});
  command_context->CmdSetPrimitiveTopology(grassland::graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  command_context->CmdDrawIndexed(36, 1, 0, 0, 0);
  command_context->CmdEndRendering();
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
