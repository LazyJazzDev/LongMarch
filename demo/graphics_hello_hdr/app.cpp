#include "app.h"

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
                                std::string(" Graphics Hello HDR"),
                            &window_);
  window_->SetHDR(true);

  std::vector<Vertex> vertices = {
      {{-0.5f, -0.1f, 0.0f}, {0.0f, 0.0f, 0.0f}},
      {{0.5f, -0.1f, 0.0f}, {3.0f, 3.0f, 3.0f}},
      {{0.5f, 0.1f, 0.0f}, {3.0f, 3.0f, 3.0f}},
      {{-0.5f, 0.1f, 0.0f}, {0.0f, 0.0f, 0.0f}},
  };
  std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

  core_->CreateBuffer(vertices.size() * sizeof(Vertex), grassland::graphics::BUFFER_TYPE_DYNAMIC, &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  core_->CreateImage(1280, 720, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);

  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "PSMain", "ps_6_0", &fragment_shader_);
  grassland::LogInfo("Shader compiled successfully");

  core_->CreateProgram({grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT},
                       grassland::graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->AddInputBinding(sizeof(Vertex), false);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, sizeof(float) * 3);
  program_->BindShader(vertex_shader_.get(), grassland::graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), grassland::graphics::SHADER_TYPE_FRAGMENT);
  program_->Finalize();
}

void Application::OnClose() {
  program_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  color_image_.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void Application::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.0, 0.0, 0.0, 1.0}});
  command_context->CmdBeginRendering({color_image_.get()}, nullptr);
  command_context->CmdBindProgram(program_.get());
  command_context->CmdBindVertexBuffers(0, {vertex_buffer_.get()}, {0});
  command_context->CmdBindIndexBuffer(index_buffer_.get(), 0);
  command_context->CmdSetViewport({0, 0, 1280, 720, 0.0f, 1.0f});
  command_context->CmdSetScissor({0, 0, 1280, 720});
  command_context->CmdSetPrimitiveTopology(grassland::graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  command_context->CmdDrawIndexed(6, 1, 0, 0, 0);
  command_context->CmdEndRendering();
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
