#include "nbody.h"

namespace {
#include "built_in_shaders.inl"
}

NBody::NBody(int n_particles) : n_particles_(n_particles) {
  graphics::Core::Settings settings;
  graphics::CreateCore(graphics::BACKEND_API_D3D12, settings, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);
  core_->CreateWindowObject(1280, 720, "NBody", false, true, &window_);
}

void NBody::Run() {
  OnInit();
  while (!glfwWindowShouldClose(window_->GLFWWindow())) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }
  core_->WaitGPU();
  OnClose();
}

void NBody::OnUpdate() {
  UpdateParticles();
  UpdateImGui();
  auto world_to_cam =
      glm::lookAt(glm::vec3{glm::vec4{10.0f, 20.0f, 30.0f, 0.0f}}, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}) *
      rotation;
  GlobalUniformObject ubo{
      glm::perspective(glm::radians(60.0f), float(window_->GetWidth()) / float(window_->GetHeight()), 0.1f, 100.0f) *
          world_to_cam,
      glm::inverse(world_to_cam), PARTICLE_SIZE, hdr_};
  global_uniform_buffer_->UploadData(&ubo, sizeof(ubo));
  particles_buffer_->UploadData(positions_.data(), sizeof(glm::vec4) * n_particles_);

  static FPSCounter fps_counter;
  window_->SetTitle("NBody fps: " + std::to_string(fps_counter.TickFPS()));
}

void NBody::OnRender() {
  std::unique_ptr<graphics::CommandContext> ctx;
  core_->CreateCommandContext(&ctx);
  ctx->CmdClearImage(frame_image_.get(), {{0.0f, 0.0f, 0.0f, 0.0f}});
  ctx->CmdBeginRendering({frame_image_.get()}, nullptr);
  ctx->CmdBindProgram(program_.get());
  ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  graphics::Scissor scissor{0, 0, window_->GetWidth(), window_->GetHeight()};
  graphics::Viewport viewport{0, 0, window_->GetWidth(), window_->GetHeight()};
  ctx->CmdSetScissor(scissor);
  ctx->CmdSetViewport(viewport);
  ctx->CmdBindVertexBuffers(0, {particles_buffer_.get()}, {0});
  ctx->CmdBindResources(0, {global_uniform_buffer_.get()});
  ctx->CmdDraw(6, n_particles_, 0, 0);
  ctx->CmdEndRendering();
  ctx->CmdBeginRendering({}, nullptr);
  ctx->CmdBindProgram(hdr_program_.get());
  ctx->CmdBindResources(0, {global_uniform_buffer_.get()});
  ctx->CmdBindResources(1, {frame_image_.get()});
  ctx->CmdDraw(6, 1, 0, 0);
  ctx->CmdEndRendering();
  ctx->CmdPresent(window_.get(), frame_image_.get());
  core_->SubmitCommandContext(ctx.get());
}

void NBody::OnInit() {
  core_->CreateImage(1280, 720, graphics::IMAGE_FORMAT_B8G8R8A8_UNORM, &frame_image_);

  window_->ResizeEvent().RegisterCallback([this](int width, int height) {
    frame_image_.reset();
    core_->CreateImage(width, height,
                       hdr_ ? graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT : graphics::IMAGE_FORMAT_B8G8R8A8_UNORM,
                       &frame_image_);
    core_->CreateProgram({frame_image_->Format()}, graphics::IMAGE_FORMAT_UNDEFINED, &program_);
    program_->SetBlendState(
        0, graphics::BlendState(graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD,
                                graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                graphics::BLEND_OP_ADD));
    program_->AddInputBinding(sizeof(glm::vec4), true);
    program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT4, 0);
    program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    program_->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program_->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program_->Finalize();
  });

  core_->CreateBuffer(sizeof(GlobalUniformObject), graphics::BUFFER_TYPE_DYNAMIC, &global_uniform_buffer_);
  core_->CreateBuffer(sizeof(glm::vec4) * n_particles_, graphics::BUFFER_TYPE_DYNAMIC, &particles_buffer_);

  positions_.resize(n_particles_);
  velocities_.resize(n_particles_);

  std::vector<glm::vec4> origins;
  std::vector<glm::vec4> initial_vels;
  for (int i = 0; i < 10; i++) {
    origins.emplace_back(RandomInSphere() * INITIAL_RADIUS * 2.0f, 0.0f);
    initial_vels.emplace_back(RandomInSphere() * INITIAL_RADIUS * 0.1f, 0.0f);
  }

  for (int i = 0; i < n_particles_; i++) {
    auto &pos = positions_[i];
    auto &vel = velocities_[i];
    int index = std::uniform_int_distribution<int>(0, origins.size() - 1)(random_device_);
    pos = glm::vec4{RandomInSphere() * INITIAL_RADIUS * 0.2f, 1.0f} + origins[index];
    vel = glm::vec4{RandomInSphere() * INITIAL_SPEED, 0.0f} + initial_vels[index];
  }

  window_->InitImGui(FileProbe::GetInstance().FindFile("fonts/simhei.ttf").c_str(), 20.0f);
  BuildRenderNode();
  window_->MouseMoveEvent().RegisterCallback([this](double xpos, double ypos) {
    ImGui::SetCurrentContext(window_->GetImGuiContext());

    static auto last_xpos = xpos;
    static auto last_ypos = ypos;
    if (glfwGetMouseButton(window_->GLFWWindow(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
      auto diffx = xpos - last_xpos;
      auto diffy = ypos - last_ypos;
      if (!ImGui::GetIO().WantCaptureMouse) {
        rotation = glm::rotate(glm::mat4{1.0f}, glm::radians(float(diffx)), glm::vec3{0.0f, 1.0f, 0.0f}) * rotation;
        rotation = glm::rotate(glm::mat4{1.0f}, glm::radians(float(diffy)), glm::vec3{1.0f, 0.0f, 0.0f}) * rotation;
      }
    }
    last_xpos = xpos;
    last_ypos = ypos;
  });
}

void NBody::OnClose() {
  program_.reset();
  particles_buffer_.reset();
  global_uniform_buffer_.reset();
}

void NBody::BuildRenderNode() {
  core_->CreateShader(GetShaderCode("shaders/particle.hlsl"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/particle.hlsl"), "PSMain", "ps_6_0", &fragment_shader_);
  core_->CreateProgram({frame_image_->Format()}, graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->SetBlendState(0, graphics::BlendState(graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE,
                                                  graphics::BLEND_OP_ADD, graphics::BLEND_FACTOR_ONE,
                                                  graphics::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, graphics::BLEND_OP_ADD));
  program_->AddInputBinding(sizeof(glm::vec4), true);
  program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT4, 0);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
  program_->Finalize();

  core_->CreateShader(GetShaderCode("shaders/hdr.hlsl"), "VSMain", "vs_6_0", &hdr_vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/hdr.hlsl"), "PSMain", "ps_6_0", &hdr_fragment_shader_);
  core_->CreateProgram({frame_image_->Format()}, graphics::IMAGE_FORMAT_UNDEFINED, &hdr_program_);
  hdr_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  hdr_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  hdr_program_->BindShader(hdr_vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  hdr_program_->BindShader(hdr_fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
  hdr_program_->Finalize();
}

float NBody::RandomFloat() {
  return std::uniform_real_distribution<float>()(random_device_);
}

glm::vec3 NBody::RandomOnSphere() {
  float z = RandomFloat() * 2.0f - 1.0f;
  float inv_z = std::sqrt(1.0f - z * z);
  float theta = RandomFloat() * glm::pi<float>() * 2.0f;
  float x = inv_z * std::sin(theta);
  float y = inv_z * std::cos(theta);
  return {x, y, z};
}

glm::vec3 NBody::RandomInSphere() {
  return RandomOnSphere() * std::pow(RandomFloat(), 0.333333333333333333f);
}

void NBody::UpdateParticles() {
#if !ENABLE_GPU
  for (int i = 0; i < n_particles_; i++) {
    auto &pos_i = positions_[i];
    for (int j = 0; j < n_particles_; j++) {
      auto &pos_j = positions_[j];
      auto diff = pos_i - pos_j;
      auto l = glm::length(diff);
      if (l < DELTA_T) {
        continue;
      }
      diff /= l * l * l;
      velocities_[i] += -diff * DELTA_T * GRAVITY_COE;
    }
  }

  for (int i = 0; i < n_particles_; i++) {
    positions_[i] += velocities_[i] * DELTA_T;
  }
#else
  UpdateStep(positions_.data(), velocities_.data(), n_particles_);
#endif
}

void NBody::UpdateImGui() {
  window_->BeginImGuiFrame();
  ImGui::SetNextWindowPos(ImVec2{0.0f, 0.0f}, ImGuiCond_Once);
  ImGui::SetNextWindowBgAlpha(0.3f);
  bool trigger_hdr_switch = false;
  if (ImGui::Begin("Statistics"), nullptr, ImGuiWindowFlags_NoMove) {
    auto current_tp = std::chrono::steady_clock::now();
    static auto last_frame_tp = current_tp;
    auto duration = current_tp - last_frame_tp;
    auto duration_ms = float(duration / std::chrono::microseconds(1)) * 1e-3f;
    ImGui::Text("Frame Duration: %.3f ms", duration_ms);
    ImGui::Text("FPS: %.3f", 1e3f / duration_ms);
    float ops = float(n_particles_) * float(n_particles_) / (duration_ms * 1e-3f);
    if (ops < 8e2f) {
      ImGui::Text("%.2f op/s", ops);
    } else if (ops < 8e5f) {
      ImGui::Text("%.2f Kop/s", ops * 1e-3f);
    } else if (ops < 8e8f) {
      ImGui::Text("%.2f Mop/s", ops * 1e-6f);
    } else {
      ImGui::Text("%.2f Gop/s", ops * 1e-9f);
    }

    if (ImGui::Button("HDR")) {
      trigger_hdr_switch = true;
    }
    ImGui::End();
    last_frame_tp = current_tp;
  }
  window_->EndImGuiFrame();
  if (trigger_hdr_switch) {
    hdr_ = !hdr_;
    window_->SetHDR(hdr_);
  }
}
