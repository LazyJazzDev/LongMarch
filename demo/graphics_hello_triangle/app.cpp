#include "app.h"

namespace {
#include "built_in_shaders.inl"
}

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{},
                                  &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  grassland::LogInfo("Device Name: {}", core_->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}",
                     core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(
      1280, 720,
      ((core_->API() == grassland::graphics::BACKEND_API_VULKAN) ? "[Vulkan]"
                                                                 : "[D3D12]") +
          std::string(" Graphics Hello Triangle"),
      &window_);

  std::vector<Vertex> vertices = {
      {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
  };
  std::vector<uint32_t> indices = {0, 1, 2};

  core_->CreateBuffer(vertices.size() * sizeof(Vertex),
                      grassland::graphics::BUFFER_TYPE_DYNAMIC,
                      &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t),
                      grassland::graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  core_->CreateImage(1280, 720,
                     grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                     &color_image_);

  if (core_->API() == grassland::graphics::BACKEND_API_VULKAN) {
    core_->CreateShader(grassland::vulkan::CompileGLSLToSPIRV(
                            GetShaderCode("shaders/vulkan/shader.vert"),
                            VK_SHADER_STAGE_VERTEX_BIT),
                        &vertex_shader_);
    core_->CreateShader(grassland::vulkan::CompileGLSLToSPIRV(
                            GetShaderCode("shaders/vulkan/shader.frag"),
                            VK_SHADER_STAGE_FRAGMENT_BIT),
                        &fragment_shader_);
    grassland::LogInfo("[Vulkan] Shader compiled successfully");
  }
#ifdef WIN32
  else if (core_->API() == grassland::graphics::BACKEND_API_D3D12) {
    core_->CreateShader(
        grassland::d3d12::CompileShader(
            GetShaderCode("shaders/d3d12/shader.hlsl"), "VSMain", "vs_6_0"),
        &vertex_shader_);
    core_->CreateShader(
        grassland::d3d12::CompileShader(
            GetShaderCode("shaders/d3d12/shader.hlsl"), "PSMain", "ps_6_0"),
        &fragment_shader_);
    grassland::LogInfo("[D3D12] Shader compiled successfully");
  }
#endif

  core_->CreateProgram({grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT},
                       grassland::graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->AddInputBinding(sizeof(Vertex), false);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3,
                              sizeof(float) * 3);
  program_->BindShader(vertex_shader_.get(),
                       grassland::graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(),
                       grassland::graphics::SHADER_TYPE_FRAGMENT);
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
  command_context->BindProgram(program_.get());
  command_context->BindVertexBuffers({vertex_buffer_.get()});
  command_context->BindIndexBuffer(index_buffer_.get());
  command_context->BindColorTargets({color_image_.get()});

  command_context->CmdSetViewport({0, 0, 1280, 720, 0.0f, 1.0f});
  command_context->CmdSetScissor({0, 0, 1280, 720});

  command_context->CmdClearImage(color_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdDrawIndexed(3, 1, 0, 0, 0);
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
