#include "nbody_cuda.h"

namespace {
#include "built_in_shaders.inl"
}

NBodyCUDA::NBodyCUDA(int n_particles) : n_particles_(n_particles) {
  graphics::Core::Settings settings;
  graphics::CreateCore(graphics::BACKEND_API_DEFAULT, settings, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);
  core_->CreateWindowObject(1920, 1080, "NBody CUDA", false, true, &window_);

  int cuda_device_index = core_->CUDADeviceIndex();
  if (cuda_device_index >= 0) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, cuda_device_index);
    LogInfo("GPU Device {}: \"{}\" with compute capability {}.{}\n", cuda_device_index, props.name, props.major,
            props.minor);
    cudaSetDevice(cuda_device_index);
  } else {
    LogError("Selected graphics device is not a CUDA device.");
  }
}

NBodyCUDA::~NBodyCUDA() {
  window_.reset();
  core_.reset();
}

void NBodyCUDA::Run() {
  if (core_->CUDADeviceIndex() == -1)
    return;
  OnInit();
  while (!glfwWindowShouldClose(window_->GLFWWindow())) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }
  core_->WaitGPU();
  OnClose();
}

void NBodyCUDA::OnUpdate() {
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

  static FPSCounter fps_counter;
  window_->SetTitle("NBody CUDA FPS: " + std::to_string(fps_counter.TickFPS()));
}

void NBodyCUDA::OnRender() {
  std::unique_ptr<graphics::CommandContext> ctx;
  core_->CreateCommandContext(&ctx);
  ctx->CmdClearImage(frame_image_.get(), {{0.0f, 0.0f, 0.0f, 0.0f}});
  ctx->CmdBeginRendering({frame_image_.get()}, nullptr);
  ctx->CmdBindProgram(program_.get());
  ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  graphics::Scissor scissor{0, 0, uint32_t(window_->GetWidth()), uint32_t(window_->GetHeight())};
  graphics::Viewport viewport{0, 0, float(window_->GetWidth()), float(window_->GetHeight())};
  ctx->CmdSetScissor(scissor);
  ctx->CmdSetViewport(viewport);
  ctx->CmdBindVertexBuffers(0, {particles_pos_.get()}, {0});
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

void NBodyCUDA::OnInit() {
  core_->CreateImage(window_->GetWidth(), window_->GetHeight(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                     &frame_image_);

  window_->ResizeEvent().RegisterCallback([this](int width, int height) {
    frame_image_.reset();
    core_->CreateImage(width, height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &frame_image_);
    core_->CreateProgram({frame_image_->Format()}, graphics::IMAGE_FORMAT_UNDEFINED, &program_);
    program_->SetBlendState(
        0, graphics::BlendState(graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE, graphics::BLEND_OP_ADD,
                                graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                graphics::BLEND_OP_ADD));
    program_->AddInputBinding(sizeof(glm::vec3), true);
    program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, 0);
    program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
    program_->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program_->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program_->Finalize();
  });

  core_->CreateBuffer(sizeof(GlobalUniformObject), graphics::BUFFER_TYPE_DYNAMIC, &global_uniform_buffer_);
  core_->CreateCUDABuffer(sizeof(glm::vec3) * n_particles_, &particles_pos_);

  particles_pos_->GetCUDAMemoryPointer(&positions_);
  cudaMalloc(&velocities_, sizeof(glm::vec3) * n_particles_);
  cudaMalloc(&positions_new_, sizeof(glm::vec3) * n_particles_);

  ResetParticles();

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

void NBodyCUDA::OnClose() {
  cudaFree(velocities_);
  cudaFree(positions_new_);

  program_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  hdr_program_.reset();
  hdr_vertex_shader_.reset();
  hdr_fragment_shader_.reset();

  particles_pos_.reset();
  global_uniform_buffer_.reset();

  frame_image_.reset();
}

void NBodyCUDA::BuildRenderNode() {
  core_->CreateShader(GetShaderCode("shaders/particle.hlsl"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/particle.hlsl"), "PSMain", "ps_6_0", &fragment_shader_);
  core_->CreateProgram({frame_image_->Format()}, graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->SetBlendState(0, graphics::BlendState(graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE,
                                                  graphics::BLEND_OP_ADD, graphics::BLEND_FACTOR_ONE,
                                                  graphics::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, graphics::BLEND_OP_ADD));
  program_->AddInputBinding(sizeof(glm::vec3), true);
  program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
  program_->Finalize();

  core_->CreateShader(GetShaderCode("shaders/hdr.hlsl"), "VSMain", "vs_6_0", &hdr_vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/hdr.hlsl"), "PSMain", "ps_6_0", &hdr_fragment_shader_);
  core_->CreateProgram({}, graphics::IMAGE_FORMAT_UNDEFINED, &hdr_program_);
  hdr_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  hdr_program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  hdr_program_->BindShader(hdr_vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  hdr_program_->BindShader(hdr_fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
  hdr_program_->Finalize();
}

float NBodyCUDA::RandomFloat() {
  return std::uniform_real_distribution<float>()(random_device_);
}

glm::vec3 NBodyCUDA::RandomOnSphere() {
  float z = RandomFloat() * 2.0f - 1.0f;
  float inv_z = std::sqrt(1.0f - z * z);
  float theta = RandomFloat() * glm::pi<float>() * 2.0f;
  float x = inv_z * std::sin(theta);
  float y = inv_z * std::cos(theta);
  return {x, y, z};
}

glm::vec3 NBodyCUDA::RandomInSphere() {
  return RandomOnSphere() * std::pow(RandomFloat(), 0.333333333333333333f);
}

void NBodyCUDA::ResetParticles() {
  std::vector<glm::vec3> positions(n_particles_);
  std::vector<glm::vec3> velocities(n_particles_);
  std::vector<glm::vec3> origins;
  std::vector<glm::vec3> initial_vels;
  for (int i = 0; i < galaxy_number_; i++) {
    origins.emplace_back(RandomInSphere() * INITIAL_RADIUS * 2.0f);
    initial_vels.emplace_back(RandomInSphere() * INITIAL_RADIUS * 0.1f);
  }

  glm::vec3 avg_vel{0.0f};
  glm::vec3 avg_pos{0.0f};
  for (int i = 0; i < galaxy_number_; i++) {
    avg_vel += initial_vels[i];
    avg_pos += origins[i];
  }
  avg_vel /= float(galaxy_number_);
  avg_pos /= float(galaxy_number_);
  for (int i = 0; i < galaxy_number_; i++) {
    initial_vels[i] -= avg_vel;
    origins[i] -= avg_pos;
  }

  for (int i = 0; i < n_particles_; i++) {
    auto &pos = positions[i];
    auto &vel = velocities[i];
    int index = std::uniform_int_distribution<int>(0, origins.size() - 1)(random_device_);
    pos =
        glm::vec3{RandomInSphere() * INITIAL_RADIUS * 0.2f * pow(10.0f / galaxy_number_, 1.0f / 3.0f)} + origins[index];
    vel = glm::vec3{RandomInSphere() * INITIAL_SPEED} + initial_vels[index];
  }

  core_->CUDABeginExecutionBarrier();
  cudaMemcpyAsync(positions_, positions.data(), sizeof(glm::vec3) * n_particles_, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(velocities_, velocities.data(), sizeof(glm::vec3) * n_particles_, cudaMemcpyHostToDevice);
  core_->CUDAEndExecutionBarrier();
}

void NBodyCUDA::UpdateParticles() {
  if (!step_)
    return;
  core_->CUDABeginExecutionBarrier();
  UpdateStep(positions_, velocities_, positions_new_, n_particles_, delta_t_);
  core_->CUDAEndExecutionBarrier();
}

void NBodyCUDA::UpdateImGui() {
  window_->BeginImGuiFrame();
  ImGui::SetNextWindowPos(ImVec2{0.0f, 0.0f}, ImGuiCond_Once);
  ImGui::SetNextWindowBgAlpha(0.3f);
  bool trigger_hdr_switch = false;
  if (ImGui::Begin("NBodyCUDA"), nullptr, ImGuiWindowFlags_NoMove) {
    ImGui::Text("Statistics");
    ImGui::Separator();
    auto current_tp = std::chrono::steady_clock::now();
    static auto last_frame_tp = current_tp;
    auto duration = current_tp - last_frame_tp;
    auto duration_ms = float(duration / std::chrono::microseconds(1)) * 1e-3f;
    ImGui::Text("Frame Duration: %.3f ms", duration_ms);
    if (step_) {
      constexpr float num_flops_per_intersection = 20.0f;  // From NVIDIA's official CUDA N-body example
      float intersection_per_second = float(n_particles_) * float(n_particles_) / (duration_ms * 1e-3f);
      float ops = intersection_per_second * num_flops_per_intersection;
      if (ops < 8e2f) {
        ImGui::Text("%.2f FLOP/s", ops);
      } else if (ops < 8e5f) {
        ImGui::Text("%.2f KFLOP/s", ops * 1e-3f);
      } else if (ops < 8e8f) {
        ImGui::Text("%.2f MFLOP/s", ops * 1e-6f);
      } else {
        ImGui::Text("%.2f GFLOP/s", ops * 1e-9f);
      }
    }

    if (ImGui::CollapsingHeader("Speed Distribution")) {
      std::vector<float> speeds(n_particles_);
      for (int i = 0; i < n_particles_; i++) {
        speeds[i] = glm::length(velocities_[i]);
      }
      std::sort(speeds.begin(), speeds.end());
      float max_speed = speeds[n_particles_ - 1];
      constexpr int num_samples = 100;
      int samples[num_samples]{};
      for (int i = 0; i < n_particles_; i++) {
        samples[std::max(std::min(int(speeds[i] / max_speed * num_samples), num_samples - 1), 0)]++;
      }
      int max_sample = 0;
      for (int i = 0; i < num_samples; i++) {
        max_sample = std::max(max_sample, samples[i]);
      }
      float normalized_samples[num_samples]{};
      for (int i = 0; i < num_samples; i++) {
        normalized_samples[i] = float(samples[i]) / float(max_sample);
      }
      ImGui::PlotLines("##1", normalized_samples, num_samples, 0, nullptr, 0.0f, 1.0f);
    }

    ImGui::NewLine();

    ImGui::Text("Control");
    ImGui::Separator();
    if (ImGui::Button("Reset")) {
      ResetParticles();
    }
    ImGui::SameLine();
    if (ImGui::Button(step_ ? "Pause" : "Resume")) {
      step_ = !step_;
    }
    // Make slider logarithmic
    ImGui::SliderFloat("Delta Time", &delta_t_, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Galaxy Number", &galaxy_number_, 1, 20, "%d");
    ImGui::NewLine();

    ImGui::Text("Visualizer");
    ImGui::Separator();
    if (ImGui::Button(("HDR: " + std::string(hdr_ ? "ON" : "OFF")).c_str())) {
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
