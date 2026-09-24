#include "nbody_cs.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/graphics/backend/metal/metal_command_context.h"
#endif

namespace {
#include "built_in_shaders.inl"
}

NBodyCS::NBodyCS(const NBodyOptions &options) : options_(options), n_particles_(options.particles) {
  random_device_.seed(options.seed);
  graphics::Core::Settings settings{2, options.debug};
  if (graphics::CreateCore(options.backend, settings, &core_) || core_->InitializeLogicalDeviceAutoSelect(false))
    throw std::runtime_error("could not initialize graphics backend");
  if (options.mode == "interactive" || options.mode == "window")
    core_->CreateWindowObject(options.width, options.height, "NBodyCS", false, true, &window_);
}

void NBodyCS::Run() {
  OnInit();
  std::cout << "Backend: " << graphics::BackendAPIString(core_->API()) << ", device: " << core_->DeviceName() << '\n';
  if (Benchmark() && options_.gpu_timing && core_->API() == graphics::BACKEND_API_VULKAN)
    profiler_ = std::make_unique<graphics::FrameProfile>(core_.get());
  std::ofstream csv;
  if (!options_.csv.empty()) {
    csv.open(options_.csv);
    if (!csv)
      throw std::runtime_error("cannot open timing CSV");
    csv << "frame,wall_ms,gpu_ms,record_ms,submit_ms,wait_ms\n" << std::fixed << std::setprecision(6);
  }

  std::vector<double> wall_times, gpu_times;
  int frame = -options_.warmup;
  while ((!window_ || !window_->ShouldClose()) && (!Benchmark() || frame < options_.frames)) {
    auto start = std::chrono::steady_clock::now();
    OnUpdate();
    OnRender();
    if (window_)
      grassland::graphics::Window::PollEvents();
    double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (Benchmark() && frame >= 0) {
      wall_times.push_back(wall);
      gpu_times.push_back(gpu_ms_);
      if (csv)
        csv << frame << ',' << wall << ',' << gpu_ms_ << ',' << record_ms_ << ',' << submit_ms_ << ',' << wait_ms_
            << '\n';
    }
    ++frame;
  }
  core_->WaitGPU();
  if (!options_.state_output.empty()) {
    std::vector<glm::vec3> positions(n_particles_), velocities(n_particles_);
    particles_pos_->DownloadData(positions.data(), positions.size() * sizeof(glm::vec3));
    particles_vel_->DownloadData(velocities.data(), velocities.size() * sizeof(glm::vec3));
    std::ofstream output(options_.state_output, std::ios::binary);
    output.write(reinterpret_cast<const char *>(positions.data()), positions.size() * sizeof(glm::vec3));
    output.write(reinterpret_cast<const char *>(velocities.data()), velocities.size() * sizeof(glm::vec3));
    if (!output)
      throw std::runtime_error("cannot write particle state");
  }
  if (!wall_times.empty()) {
    auto mean = [](const std::vector<double> &v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    std::cout << std::fixed << std::setprecision(6) << "RESULT {\"backend\":\""
              << graphics::BackendAPIString(core_->API()) << "\",\"mode\":\"" << options_.mode
              << "\",\"particles\":" << n_particles_ << ",\"frames\":" << wall_times.size()
              << ",\"warmup\":" << options_.warmup << ",\"seed\":" << options_.seed << ",\"width\":" << options_.width
              << ",\"height\":" << options_.height << ",\"wall_ms\":" << mean(wall_times)
              << ",\"gpu_ms\":" << mean(gpu_times) << ",\"fps_equivalent\":" << 1000.0 / mean(wall_times) << "}\n";
  }
  if (window_)
    window_->TerminateImGui();
  OnClose();
}

void NBodyCS::OnUpdate() {
  if (window_)
    UpdateImGui();
  auto world_to_cam =
      glm::lookAt(glm::vec3{glm::vec4{10.0f, 20.0f, 30.0f, 0.0f}}, glm::vec3{0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}) *
      rotation;
  GlobalUniformObject ubo{glm::perspective(glm::radians(60.0f),
                                           float(frame_image_ ? frame_image_->Extent().width : options_.width) /
                                               float(frame_image_ ? frame_image_->Extent().height : options_.height),
                                           0.1f, 100.0f) *
                              world_to_cam,
                          glm::inverse(world_to_cam), PARTICLE_SIZE, hdr_};
  global_uniform_buffer_->UploadData(&ubo, sizeof(ubo));

  NBodyGlobalSettings global_settings;
  global_settings.delta_t = delta_t_;
  global_settings.gravity = GRAVITY_COE;
  global_settings.num_particle = n_particles_;
  global_settings_buffer_->UploadData(&global_settings, sizeof(global_settings));

  static FPSCounter fps_counter;
  if (window_ && !Benchmark())
    window_->SetTitle("NBody Compute Shader FPS: " + std::to_string(fps_counter.TickFPS()));
}

void NBodyCS::OnRender() {
  using Clock = std::chrono::steady_clock;
  auto start = Clock::now();
  if (profiler_)
    profiler_->Begin();
  std::unique_ptr<graphics::CommandContext> ctx;
  core_->CreateCommandContext(&ctx);
  int gpu_scope = profiler_ ? profiler_->BeginGpu(ctx.get(), "frame") : -1;
  if (step_) {
    ctx->CmdBindComputeProgram(nbody_compute_program_.get());
    ctx->CmdBindResources(0, {particles_pos_.get()}, graphics::BIND_POINT_COMPUTE);
    ctx->CmdBindResources(1, {particles_vel_.get()}, graphics::BIND_POINT_COMPUTE);
    ctx->CmdBindResources(2, {particles_pos_new_.get()}, graphics::BIND_POINT_COMPUTE);
    ctx->CmdBindResources(3, {global_settings_buffer_.get()}, graphics::BIND_POINT_COMPUTE);
    ctx->CmdDispatch(n_particles_ / 128, 1, 1);
    ctx->CmdCopyBuffer(particles_pos_.get(), particles_pos_new_.get(), particles_pos_->Size());
  }
  if (options_.mode != "compute") {
    ctx->CmdClearImage(frame_image_.get(), {{0.0f, 0.0f, 0.0f, 0.0f}});
    ctx->CmdBeginRendering({frame_image_.get()}, nullptr);
    ctx->CmdBindProgram(program_.get());
    ctx->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    graphics::Scissor scissor{0, 0, frame_image_->Extent().width, frame_image_->Extent().height};
    graphics::Viewport viewport{0, 0, float(frame_image_->Extent().width), float(frame_image_->Extent().height)};
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
  }
  if (window_)
    ctx->CmdPresent(window_.get(), frame_image_.get());
  if (profiler_)
    profiler_->EndGpu(ctx.get(), gpu_scope);
  auto recorded = Clock::now();
  core_->SubmitCommandContext(ctx.get());
  auto submitted = Clock::now();
  if (Benchmark()) {
    core_->WaitGPU();
    auto completed = Clock::now();
    record_ms_ = std::chrono::duration<double, std::milli>(recorded - start).count();
    submit_ms_ = std::chrono::duration<double, std::milli>(submitted - recorded).count();
    wait_ms_ = std::chrono::duration<double, std::milli>(completed - submitted).count();
    gpu_ms_ = -1;
    if (profiler_) {
      profiler_->Finish();
      gpu_ms_ = profiler_->gpu_ms.at("frame");
    }
#if defined(LONGMARCH_METAL_ENABLED)
    if (options_.gpu_timing && core_->API() == graphics::BACKEND_API_METAL) {
      auto native = dynamic_cast<graphics::backend::MetalCommandContext *>(ctx.get())->Handle();
      gpu_ms_ = (native->GPUEndTime() - native->GPUStartTime()) * 1000.0;
      if (gpu_ms_ <= 0)
        throw std::runtime_error("Metal GPU timestamps unavailable");
    }
#endif
  }
}

void NBodyCS::OnInit() {
  if (options_.mode != "compute") {
    const auto size = window_ ? window_->GetFramebufferSize() : glm::ivec2{options_.width, options_.height};
    core_->CreateImage(size.x, size.y, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &frame_image_);
  }

  if (window_)
    window_->FramebufferResizeEvent().RegisterCallback([this](int width, int height) {
      core_->WaitGPU();
      if (width <= 0 || height <= 0)
        return;
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
      program_->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_PIXEL);
      program_->Finalize();
    });

  core_->CreateBuffer(sizeof(GlobalUniformObject), graphics::BUFFER_TYPE_DYNAMIC, &global_uniform_buffer_);
  core_->CreateBuffer(sizeof(glm::vec3) * n_particles_, graphics::BUFFER_TYPE_STATIC, &particles_pos_);
  core_->CreateBuffer(sizeof(glm::vec3) * n_particles_, graphics::BUFFER_TYPE_STATIC, &particles_vel_);
  core_->CreateBuffer(sizeof(glm::vec3) * n_particles_, graphics::BUFFER_TYPE_STATIC, &particles_pos_new_);

  core_->CreateBuffer(sizeof(NBodyGlobalSettings), graphics::BUFFER_TYPE_DYNAMIC, &global_settings_buffer_);

  ResetParticles();

  if (window_) {
    window_->InitImGui(FileProbe::GetInstance().FindFile("fonts/simhei.ttf").c_str(), 20.0f);
    if (Benchmark()) {
      ImGui::SetCurrentContext(window_->GetImGuiContext());
      ImGui::GetIO().IniFilename = nullptr;
    }
  }

  BuildRenderNode();
  if (window_)
    window_->MouseMoveEvent().RegisterCallback([this](double xpos, double ypos) {
      ImGui::SetCurrentContext(window_->GetImGuiContext());

      if (!cursor_initialized_) {
        last_cursor_ = {xpos, ypos};
        cursor_initialized_ = true;
      }
      if (window_->IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT)) {
        auto diffx = xpos - last_cursor_.x;
        auto diffy = ypos - last_cursor_.y;
        if (!ImGui::GetIO().WantCaptureMouse) {
          rotation = glm::rotate(glm::mat4{1.0f}, glm::radians(float(diffx)), glm::vec3{0.0f, 1.0f, 0.0f}) * rotation;
          rotation = glm::rotate(glm::mat4{1.0f}, glm::radians(float(diffy)), glm::vec3{1.0f, 0.0f, 0.0f}) * rotation;
        }
      }
      last_cursor_ = {xpos, ypos};
    });
}

void NBodyCS::OnClose() {
  nbody_compute_program_.reset();
  nbody_compute_shader_.reset();

  hdr_program_.reset();
  hdr_vertex_shader_.reset();
  hdr_fragment_shader_.reset();

  program_.reset();
  fragment_shader_.reset();
  vertex_shader_.reset();

  particles_pos_.reset();
  particles_vel_.reset();
  particles_pos_new_.reset();
  global_uniform_buffer_.reset();
}

void NBodyCS::BuildRenderNode() {
  core_->CreateShader(GetShaderCode("shaders/nbody.hlsl"), "CSMain", "cs_6_0", &nbody_compute_shader_);
  core_->CreateComputeProgram(nbody_compute_shader_.get(), &nbody_compute_program_);
  nbody_compute_program_->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
  nbody_compute_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  nbody_compute_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  nbody_compute_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  nbody_compute_program_->Finalize();
  if (options_.mode == "compute")
    return;

  core_->CreateShader(GetShaderVirtualFileSystem(), "shaders/particle.hlsl", "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderVirtualFileSystem(), "shaders/particle.hlsl", "PSMain", "ps_6_0", &fragment_shader_);
  core_->CreateProgram({frame_image_->Format()}, graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->SetBlendState(0, graphics::BlendState(graphics::BLEND_FACTOR_ONE, graphics::BLEND_FACTOR_ONE,
                                                  graphics::BLEND_OP_ADD, graphics::BLEND_FACTOR_ONE,
                                                  graphics::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, graphics::BLEND_OP_ADD));
  program_->AddInputBinding(sizeof(glm::vec3), true);
  program_->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_PIXEL);
  program_->Finalize();

  core_->CreateShader(GetShaderVirtualFileSystem(), "shaders/hdr.hlsl", "VSMain", "vs_6_0", &hdr_vertex_shader_);
  core_->CreateShader(GetShaderVirtualFileSystem(), "shaders/hdr.hlsl", "PSMain", "ps_6_0", &hdr_fragment_shader_);
  core_->CreateProgram({}, graphics::IMAGE_FORMAT_UNDEFINED, &hdr_program_);
  hdr_program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  hdr_program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  hdr_program_->BindShader(hdr_vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
  hdr_program_->BindShader(hdr_fragment_shader_.get(), graphics::SHADER_TYPE_PIXEL);
  hdr_program_->Finalize();
}

float NBodyCS::RandomFloat() {
  return std::uniform_real_distribution<float>()(random_device_);
}

glm::vec3 NBodyCS::RandomOnSphere() {
  float z = RandomFloat() * 2.0f - 1.0f;
  float inv_z = std::sqrt(1.0f - z * z);
  float theta = RandomFloat() * glm::pi<float>() * 2.0f;
  float x = inv_z * std::sin(theta);
  float y = inv_z * std::cos(theta);
  return {x, y, z};
}

glm::vec3 NBodyCS::RandomInSphere() {
  return RandomOnSphere() * std::pow(RandomFloat(), 0.333333333333333333f);
}

void NBodyCS::ResetParticles() {
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

  std::vector<glm::vec3> positions(n_particles_);
  std::vector<glm::vec3> velocities(n_particles_);

  for (int i = 0; i < n_particles_; i++) {
    auto &pos = positions[i];
    auto &vel = velocities[i];
    int index = std::uniform_int_distribution<int>(0, origins.size() - 1)(random_device_);
    pos =
        glm::vec3{RandomInSphere() * INITIAL_RADIUS * 0.2f * pow(10.0f / galaxy_number_, 1.0f / 3.0f)} + origins[index];
    vel = glm::vec3{RandomInSphere() * INITIAL_SPEED} + initial_vels[index];
  }

  particles_pos_->UploadData(positions.data(), sizeof(glm::vec3) * n_particles_);
  particles_vel_->UploadData(velocities.data(), sizeof(glm::vec3) * n_particles_);
}

void NBodyCS::UpdateImGui() {
  window_->BeginImGuiFrame();
  ImGui::SetNextWindowPos(ImVec2{0.0f, 0.0f}, ImGuiCond_Once);
  ImGui::SetNextWindowBgAlpha(0.3f);
  bool trigger_hdr_switch = false;
  if (ImGui::Begin("NBodyCS", nullptr, ImGuiWindowFlags_NoMove)) {
    ImGui::Text("Statistics");
    ImGui::Separator();
    ImGui::Text("Backend: %s", graphics::BackendAPIString(core_->API()));
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
      std::vector<glm::vec3> velocities(n_particles_);
      particles_vel_->DownloadData(velocities.data(), sizeof(glm::vec3) * n_particles_);
      std::vector<float> speeds(n_particles_);
      for (int i = 0; i < n_particles_; i++) {
        speeds[i] = glm::length(velocities[i]);
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
    last_frame_tp = current_tp;
  }

  ImGui::End();
  window_->EndImGuiFrame();
  if (trigger_hdr_switch) {
    hdr_ = !hdr_;
    window_->SetHDR(hdr_);
  }
}
