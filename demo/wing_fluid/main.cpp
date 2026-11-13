#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "imgui.h"
#include "simulation.h"

namespace {
using namespace grassland::rhi;
#include "display.slang.h"

bool CUDA(cudaError_t error) {
  if (error == cudaSuccess)
    return true;
  std::cerr << "CUDA: " << cudaGetErrorString(error) << '\n';
  return false;
}

struct Runtime {
  cudaStream_t stream{};
  std::unique_ptr<wing_fluid::Simulation> simulation;
  ObjPtr<ICUDABuffer> field;
  ObjPtr<ICUDATimeline> timeline;
  ObjPtr<ISubmissionToken> last;
  uint64_t serial{};

  ~Runtime() {
    if (stream)
      cudaStreamSynchronize(stream);
    if (last)
      last->WaitHost();
    simulation.reset();
    if (stream)
      cudaStreamDestroy(stream);
  }
};

struct Options {
  BackendAPI backend{BACKEND_API_DEFAULT};
  int width{512}, frames{}, steps{4}, view{};
  bool headless{}, validate{}, debug{};
  std::string capture;
  wing_fluid::Parameters parameters;
};

struct LoadDisplay {
  wing_fluid::LoadCoefficients value;
  std::vector<float> x, y, moment;
  float sample_time{};

  void Update(const wing_fluid::LoadCoefficients &load, float dt) {
    if (!load.valid) {
      value = load;
      return;
    }
    float a = value.valid ? 1 - std::exp(-dt / 0.2f) : 1;
    value.x += a * (load.x - value.x);
    value.y += a * (load.y - value.y);
    value.moment += a * (load.moment - value.moment);
    value.angle = load.angle;
    value.valid = true;
    sample_time += dt;
    if (sample_time < 0.05f && !x.empty())
      return;
    sample_time = 0;
    for (auto *v : {&x, &y, &moment})
      if (v->size() == 240)
        v->erase(v->begin());
    x.push_back(value.x);
    y.push_back(value.y);
    moment.push_back(value.moment);
  }

  void Draw(bool paused) {
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("Loads", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
    ImGui::TextUnformatted("FLUID LOAD / LIVE TRENDS");
    ImGui::TextDisabled(paused ? "Paused" : "Smoothed over 0.2 simulation seconds");
    auto *draw = ImGui::GetWindowDrawList();
    auto origin = ImGui::GetCursorScreenPos();
    float width = ImGui::GetContentRegionAvail().x;
    ImVec2 center{origin.x + width * 0.5f, origin.y + 145};
    auto point = [&](float x, float y) { return ImVec2{center.x + x, center.y - y}; };
    auto arrow = [&](ImVec2 a, ImVec2 b, ImU32 color) {
      float dx = b.x - a.x, dy = b.y - a.y, length = std::hypot(dx, dy);
      if (length < 1)
        return;
      dx /= length;
      dy /= length;
      draw->AddLine(a, b, color, 3);
      draw->AddTriangleFilled(b, {b.x - 10 * dx + 5 * dy, b.y - 10 * dy - 5 * dx},
                              {b.x - 10 * dx - 5 * dy, b.y - 10 * dy + 5 * dx}, color);
    };
    draw->AddLine(point(-width * 0.4f, 0), point(width * 0.4f, 0), IM_COL32(85, 95, 110, 255));
    draw->AddLine(point(0, -120), point(0, 120), IM_COL32(85, 95, 110, 255));
    draw->AddText(point(width * 0.4f - 20, -8), IM_COL32_WHITE, "+x");
    draw->AddText(point(5, 120), IM_COL32_WHITE, "+y");
    float angle = value.angle * 0.01745329252f;
    ImVec2 outline[82];
    for (int i = 0; i < 82; ++i) {
      float t = (i <= 40 ? i : 81 - i) / 40.0f;
      float h =
          0.6f *
          (0.2969f * std::sqrt(t) - 0.126f * t - 0.3516f * t * t + 0.2843f * t * t * t - 0.1036f * t * t * t * t) *
          (i <= 40 ? 1 : -1);
      float x = (t - 0.5f) * 160, y = h * 160;
      outline[i] = point(x * std::cos(angle) + y * std::sin(angle), -x * std::sin(angle) + y * std::cos(angle));
    }
    draw->AddPolyline(outline, 82, IM_COL32(190, 200, 215, 255), ImDrawFlags_Closed, 2);
    const auto cyan = IM_COL32(65, 225, 200, 255), orange = IM_COL32(255, 170, 65, 255);
    float magnitude = std::hypot(value.x, value.y);
    // Fixed nonlinear scale preserves direction and relative changes without clipping transients.
    float scale = 115 / (1 + magnitude);
    arrow(center, point(value.x * scale, value.y * scale), cyan);
    if (std::abs(value.moment) > 0.0001f) {
      float arc = 5.5f * std::abs(value.moment) / (0.1f + std::abs(value.moment));
      float sign = value.moment > 0 ? 1 : -1;
      ImVec2 last = point(100, 0);
      for (int i = 1; i <= 40; ++i) {
        float a = sign * arc * i / 40;
        auto next = point(100 * std::cos(a), 100 * std::sin(a));
        if (i == 40)
          arrow(last, next, orange);
        else
          draw->AddLine(last, next, orange, 3);
        last = next;
      }
    }
    ImGui::Dummy({width, 290});
    ImGui::TextColored({0.25f, 0.9f, 0.8f, 1}, "Force: downstream / upward");
    ImGui::TextColored({1, 0.67f, 0.25f, 1}, "Moment: %s",
                       value.moment >= 0 ? "counterclockwise (+)" : "clockwise (-)");
    if (!value.valid)
      ImGui::TextUnformatted("Waiting for a simulation step...");
    ImGui::TextDisabled("Fixed compressed arrow scale; zero at center");
    ImGui::Separator();
    auto plot = [&](const char *label, const std::vector<float> &samples) {
      float range = 0.1f;
      for (float v : samples)
        range = std::max(range, std::abs(v) * 1.1f);
      ImGui::PlotLines(label, samples.data(), int(samples.size()), 0, nullptr, -range, range, {width - 55, 65});
    };
    plot("Fx", x);
    plot("Fy", y);
    plot("Mz", moment);
    ImGui::TextWrapped(
        "Pressure + approximate wall shear. Moment about mid-chord. Curves auto-scale around zero; qualitative trends only.");
    ImGui::End();
  }
};

bool Parse(int argc, char **argv, Options &o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--headless")
      o.headless = true;
    else if (a == "--validate") {
      o.validate = true;
      o.headless = true;
    } else if (a == "--debug")
      o.debug = true;
    else if (i + 1 < argc) {
      std::string value = argv[++i];
      if (a == "--backend") {
        if (value == "vulkan")
          o.backend = BACKEND_API_VULKAN;
        else if (value == "d3d12")
          o.backend = BACKEND_API_D3D12;
        else
          return false;
      } else if (a == "--grid")
        o.width = std::atoi(value.c_str());
      else if (a == "--frames")
        o.frames = std::atoi(value.c_str());
      else if (a == "--steps")
        o.steps = std::atoi(value.c_str());
      else if (a == "--speed")
        o.parameters.speed = std::atof(value.c_str());
      else if (a == "--angle")
        o.parameters.angle = std::atof(value.c_str());
      else if (a == "--viscosity")
        o.parameters.viscosity = std::atof(value.c_str());
      else if (a == "--view") {
        if (value == "dye")
          o.view = 0;
        else if (value == "speed")
          o.view = 1;
        else if (value == "vorticity")
          o.view = 2;
        else if (value == "pressure")
          o.view = 3;
        else
          return false;
      } else if (a == "--capture")
        o.capture = value;
      else
        return false;
    } else
      return false;
  }
  return o.width >= 128 && o.width <= 1024 && !(o.width & (o.width - 1)) && o.steps >= 1 && o.steps <= 16 &&
         o.frames >= 0 && o.parameters.speed >= 0.1f && o.parameters.speed <= 3 && o.parameters.angle >= -25 &&
         o.parameters.angle <= 25 && std::isfinite(o.parameters.angle) && o.parameters.viscosity >= 0.0005f &&
         o.parameters.viscosity <= 0.01f;
}

bool Render(IDevice *device, Runtime &runtime, Options options) {
  using wing_fluid::Simulation;
  auto &sim = *runtime.simulation;
  auto parameters = options.parameters;
  const int width = options.width, height = width / 2;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IWindow> window;
  ObjPtr<IWindow> load_window;
  LoadDisplay loads;
  bool show_loads = true;
  if (!options.headless) {
    window = device->CreateGLFWWindow({1440, 860, "Wing Fluid | CUDA + RHI"});
    if (!window || !window->EnableImGui(true, LONGMARCH_ASSETS_DIR "fonts/simhei.ttf", 19))
      return false;
  }
  ObjPtr<IBuffer> uniform = device->CreateBuffer({256});
  BindingLayoutDesc layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                            {1, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  if (!group || !uniform || !group->BindResource(0, uniform.Get()) || !group->BindResource(1, runtime.field.Get()))
    return false;
  std::string source(reinterpret_cast<const char *>(display_slang), display_slang_len);
  ObjPtr<IShader> vs = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> ps = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.pipeline_layout.groups = {layout};
  desc.color_formats = {IMAGE_FORMAT_R8G8B8A8_UNORM};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  if (!vs || !ps || !program)
    return false;
  ObjPtr<IImage> canvas;
  int mode = options.view;
  float dye_strength = 1.8f, elapsed = 0, fps = 0, time_scale = 1;
  bool paused = false, reset = false;
  wing_fluid::Diagnostics diagnostics;
  auto previous = std::chrono::steady_clock::now();
  auto beginning = previous;
  const int limit = options.frames ? options.frames : options.headless ? 120 : 0;
  for (int frame = 0; !limit || frame < limit; ++frame) {
    if (window && (!PollEvents() || window->ShouldClose()))
      break;
    // Bound CPU submissions; the GPU interop edges below establish memory/ownership ordering.
    if (runtime.last && !runtime.last->WaitHost())
      return false;
    auto now = std::chrono::steady_clock::now();
    float seconds = std::chrono::duration<float>(now - previous).count();
    previous = now;
    if (seconds > 0)
      fps = fps == 0 ? 1 / seconds : fps * 0.9f + 0.1f / seconds;
    uint32_t sw = 1440, sh = 860;
    ObjPtr<IImage> swap;
    if (window) {
      if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
        return false;
      swap = window->GetNextImage();
      if (!swap) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        --frame;
        continue;
      }
      sw = swap->GetDesc().width;
      sh = swap->GetDesc().height;
      if (!window->UpdateImGui([&] {
            auto &style = ImGui::GetStyle();
            style.WindowRounding = 10;
            style.FrameRounding = 5;
            style.GrabRounding = 5;
            style.Colors[ImGuiCol_WindowBg] = {0.035f, 0.05f, 0.075f, 1};
            style.Colors[ImGuiCol_FrameBg] = {0.08f, 0.14f, 0.19f, 1};
            style.Colors[ImGuiCol_SliderGrab] = {0.15f, 0.65f, 0.58f, 1};
            style.Colors[ImGuiCol_SliderGrabActive] = {0.3f, 0.9f, 0.75f, 1};
            auto size = ImGui::GetIO().DisplaySize;
            ImGui::SetNextWindowPos({size.x - 360, 0});
            ImGui::SetNextWindowSize({360, size.y});
            ImGui::Begin("Controls", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextColored({0.35f, 0.9f, 0.8f, 1}, "WING / FLUID");
            ImGui::TextUnformatted("2D incompressible Navier-Stokes");
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextUnformatted("NACA 0012 / symmetric section");
            ImGui::Text("Chord %.2f m | channel 6 x 3 m", Simulation::chord);
            ImGui::Spacing();
            ImGui::PushItemWidth(-1);
            ImGui::TextUnformatted("Flow speed");
            ImGui::SliderFloat("##speed", &parameters.speed, 0.1f, 3.0f, "%.2f m/s");
            ImGui::TextUnformatted("Angle of attack");
            ImGui::SliderFloat("##angle", &parameters.angle, -25, 25, "%.1f deg");
            ImGui::Text("Reynolds number: %.0f", parameters.speed * Simulation::chord / parameters.viscosity);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextUnformatted("Display");
            ImGui::Combo("##view", &mode, "Dye streaks\0Speed\0Vorticity\0Pressure\0");
            if (mode == 0)
              ImGui::SliderFloat("##contrast", &dye_strength, 0.5f, 4.0f, "Dye contrast %.1f");
            else if (mode == 1)
              ImGui::TextWrapped("Speed: dark = 0, orange = %.1f m/s", parameters.speed * 2);
            else if (mode == 2)
              ImGui::TextWrapped("Blue: clockwise / orange: counterclockwise");
            else
              ImGui::TextWrapped("Relative to outlet: blue negative, orange positive");
            ImGui::Checkbox("Pause", &paused);
            ImGui::SameLine();
            if (ImGui::Button("Reset flow"))
              reset = true;
            ImGui::Checkbox("Load trends window", &show_loads);
            ImGui::SliderFloat("##time", &time_scale, 0.1f, 2, "Simulation speed %.1fx");
            if (ImGui::CollapsingHeader("Solver settings")) {
              ImGui::TextUnformatted("Kinematic viscosity");
              ImGui::SliderFloat("##viscosity", &parameters.viscosity, 0.0005f, 0.01f, "%.4f m2/s",
                                 ImGuiSliderFlags_Logarithmic);
            }
            ImGui::PopItemWidth();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("%.1f FPS | simulation %.2f s", fps, elapsed);
            ImGui::Text("MAC grid: %d x %d", width, height);
            ImGui::Text("Multigrid: %d levels, %d V-cycles", sim.Levels(), parameters.cycles);
            ImGui::Text("Divergence RMS: %.3g", diagnostics.divergence_after);
            ImGui::Text("Before projection: %.3g", diagnostics.divergence_before);
            ImGui::Text("Max speed: %.3f m/s", diagnostics.max_speed);
            ImGui::Spacing();
            ImGui::TextWrapped("CUDA solves advection, viscosity and pressure. RHI renders the shared device buffer.");
            ImGui::Spacing();
            ImGui::TextDisabled("Exploratory 2D laminar model");
            ImGui::TextWrapped(
                "Stair-step stationary wing; no turbulence model or engineering lift/drag prediction. Changing angle revoxelizes the obstacle.");
            ImGui::End();
            ImGui::SetNextWindowPos({24, 20});
            ImGui::Begin("Title", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs |
                             ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextColored({0.35f, 0.9f, 0.8f, 1}, "WIND TUNNEL");
            ImGui::Text("U = %.2f m/s    incidence = %+.1f deg", parameters.speed, parameters.angle);
            ImGui::End();
          }))
        return false;
    }
    if (runtime.last && !runtime.timeline->Wait(runtime.stream, {runtime.last->GetEvent()}))
      return false;
    if (reset) {
      if (!sim.Reset(parameters))
        return false;
      reset = false;
      elapsed = 0;
      loads = {};
    }
    float dx = Simulation::domain_width / width;
    float dt = std::min(0.65f * dx / std::max({3 * parameters.speed, diagnostics.max_speed * 1.25f, 0.1f}),
                        0.2f * dx * dx / parameters.viscosity);
    int steps = options.steps;
    if (window) {
      float duration = std::min(seconds, 0.05f) * time_scale;
      steps = std::max(1, std::min(32, int(std::ceil(duration / dt))));
      dt = std::min(dt, duration / steps);
    }
    if (!paused)
      for (int step = 0; step < steps; ++step) {
        if (!sim.Step(parameters, dt))
          return false;
        elapsed += dt;
      }
    if (frame % 20 == 0 || frame + 1 == limit) {
      if (!sim.Measure(diagnostics) || !diagnostics.finite)
        return false;
    }
    wing_fluid::LoadCoefficients load;
    if (!sim.MeasureLoads(load))
      return false;
    if (!paused || !load.valid)
      loads.Update(load, paused ? 0 : steps * dt);
    ObjPtr<IImage> load_image;
    if (window) {
      if (load_window && load_window->ShouldClose())
        show_loads = false;
      if (!show_loads)
        load_window.Release();
      if (show_loads && !load_window) {
        load_window = device->CreateGLFWWindow({560, 740, "Wing Fluid | Load trends"});
        if (!load_window || !load_window->EnableImGui(true, LONGMARCH_ASSETS_DIR "fonts/simhei.ttf", 19))
          return false;
        int x, y, w, h, mx, my, mw, mh;
        glfwGetWindowPos(window->GLFWWindow(), &x, &y);
        glfwGetWindowSize(window->GLFWWindow(), &w, &h);
        glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), &mx, &my, &mw, &mh);
        glfwSetWindowPos(load_window->GLFWWindow(), std::max(mx, std::min(x + w + 12, mx + mw - 560)),
                         std::max(my + 30, std::min(y, my + mh - 740)));
      }
      if (load_window) {
        if (load_window->IsSwapChainInvalid() && !load_window->RebuildSwapChain())
          return false;
        load_image = load_window->GetNextImage();
        if (load_image && !load_window->UpdateImGui([&] { loads.Draw(paused); }))
          return false;
      }
    }
    if (!sim.Export(runtime.field->GetDevicePointer()))
      return false;
    Event ready = runtime.timeline->Signal(runtime.stream, ++runtime.serial);
    if (!ready.timeline)
      return false;
    if (!canvas || canvas->GetDesc().width != sw || canvas->GetDesc().height != sh) {
      canvas = device->CreateImage({sw, sh, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!canvas)
        return false;
    }

    struct Settings {
      uint32_t width, height, mode;
      float speed, contrast, padding[3];
    } settings{uint32_t(width), uint32_t(height), uint32_t(mode), parameters.speed, dye_strength, {0, 0, 0}};

    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands || !commands->CmdUploadBuffer(uniform.Get(), 0, &settings, sizeof(settings)) ||
        !commands->CmdBindProgram(program.Get()) || !commands->CmdBindGroup(0, group.Get()) ||
        !commands->CmdBeginRendering(
            {{{canvas.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.012f, 0.018f, 0.035f, 1}}}}))
      return false;
    int logical_width = int(sw), logical_height = int(sh);
    if (window)
      glfwGetWindowSize(window->GLFWWindow(), &logical_width, &logical_height);
    float panel = window ? 360.0f * sw / std::max(logical_width, 1) : 0;
    float vw = std::max(1.0f, sw - panel), vh = std::min(float(sh), vw * 0.5f);
    vw = std::min(vw, vh * 2);
    float left = (sw - panel - vw) * 0.5f, top = (sh - vh) * 0.5f;
    if (!commands->CmdSetViewport({left, top, vw, vh, 0, 1}) || !commands->CmdSetScissor({0, 0, sw, sh}) ||
        !commands->CmdDraw(3) || !commands->CmdEndRendering())
      return false;
    if (window && (!commands->CmdDrawImGui(window.Get(), canvas.Get()) ||
                   !commands->CmdPresentImage(window.Get(), canvas.Get(), COLOR_SPACE_BT709_LINEAR)))
      return false;
    ObjPtr<IDownloadToken> download;
    if (load_image && (!commands->CmdClearImage(load_image.Get(), {0.035f, 0.05f, 0.075f, 1.0f}) ||
                       !commands->CmdPresent(load_window.Get())))
      return false;
    if (!options.capture.empty() && frame + 1 == limit) {
      download = commands->CmdDownloadImage(canvas.Get());
      if (!download)
        return false;
    }
    SubmitInfo info;
    info.waits = {ready};
    runtime.last = commands->Submit(info);
    if (!runtime.last)
      return false;
    if (download) {
      std::vector<unsigned char> pixels(size_t(sw) * sh * 4);
      if (!download->WaitHost() || !download->CopyTo(pixels.data(), pixels.size()))
        return false;
      std::ofstream file(options.capture, std::ios::binary);
      file << "P6\n" << sw << ' ' << sh << "\n255\n";
      for (size_t i = 0; i < pixels.size(); i += 4)
        file.write(reinterpret_cast<char *>(pixels.data() + i), 3);
      if (!file)
        return false;
    }
  }
  if (runtime.last && !runtime.last->WaitHost())
    return false;
  std::cout << "simulation_time=" << elapsed
            << " wall_time=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - beginning).count()
            << " div_before=" << diagnostics.divergence_before << " div_after=" << diagnostics.divergence_after
            << " max_speed=" << diagnostics.max_speed << " solid_speed=" << diagnostics.solid_speed << '\n';
  wing_fluid::LoadCoefficients final_load;
  if (!sim.MeasureLoads(final_load))
    return false;
  std::cout << "load_coefficients x=" << final_load.x << " y=" << final_load.y << " moment=" << final_load.moment
            << '\n';
  return true;
}
}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!Parse(argc, argv, options)) {
    std::cerr << "Usage: demo_wing_fluid [--backend vulkan|d3d12] [--grid 128|256|512|1024] [--speed 0.1..3] "
                 "[--angle -25..25] [--viscosity 0.0005..0.01] [--view dye|speed|vorticity|pressure] "
                 "[--frames N] [--steps 1..16] [--headless] [--validate] [--debug] [--capture image.ppm]\n";
    return 1;
  }
  ObjPtr<ICore> core = CreateCore(options.backend, {options.debug});
  if (!core)
    return 1;
  ObjPtr<IDevice> device;
  auto physical = core->GetPhysicalDevices();
  for (size_t i = 0; i < physical.size(); ++i)
    if (physical[i].cuda_device >= 0) {
      device = core->CreateDevice(int(i));
      if (device)
        break;
    }
  if (!device) {
    std::cerr << "A CUDA-capable RHI device is required.\n";
    return 1;
  }
  if (!CUDA(cudaSetDevice(device->GetCUDADeviceIndex())))
    return 1;
  Runtime runtime;
  if (!CUDA(cudaStreamCreateWithFlags(&runtime.stream, cudaStreamNonBlocking)))
    return 1;
  runtime.simulation = std::make_unique<wing_fluid::Simulation>();
  auto &sim = *runtime.simulation;
  if (!sim.Initialize(options.width, options.width / 2, runtime.stream) || !sim.Reset(options.parameters)) {
    std::cerr << sim.Error() << '\n';
    return 1;
  }
  runtime.field = device->CreateCUDABuffer({uint64_t(options.width) * (options.width / 2) * 16});
  runtime.timeline = device->CreateCUDATimeline();
  if (!runtime.field || !runtime.timeline)
    return 1;
  std::cout << device->Name() << " | " << options.width << 'x' << options.width / 2 << " | " << sim.Levels()
            << " levels\n";
  if (options.validate) {
    wing_fluid::Diagnostics initial;
    if (!sim.Measure(initial))
      return 2;
    std::cout << "reset_projection before=" << initial.divergence_before << " after=" << initial.divergence_after
              << '\n';
    if (!initial.finite || initial.solid_speed != 0 || initial.divergence_after > 0.001f)
      return 2;
    float residual, error;
    if (!sim.ValidateLoads(error))
      return 1;
    std::cout << "surface_load relative_error=" << error << '\n';
    if (!std::isfinite(error) || error > 0.00001f)
      return 2;
    if (!sim.ValidatePressure(residual, error))
      return 1;
    std::cout << "manufactured_pressure relative_residual=" << residual << " relative_error=" << error << '\n';
    if (!std::isfinite(error) || residual > 0.01f || error > 0.05f)
      return 2;
    if (!sim.Reset(options.parameters))
      return 1;
    // Exercise sudden inlet/geometry changes without resetting the flow between cases.
    for (auto values : {std::pair<float, float>{1, 0}, {3, 25}, {0.2f, -25}, {1, 8}}) {
      auto p = options.parameters;
      p.speed = values.first;
      p.angle = values.second;
      float dx = wing_fluid::Simulation::domain_width / options.width;
      float dt = std::min(0.5f * dx / 10.0f, 0.2f * dx * dx / p.viscosity);
      for (int step = 0; step < 96; ++step)
        if (!sim.Step(p, dt))
          return 1;
      wing_fluid::Diagnostics d;
      if (!sim.Measure(d))
        return 1;
      std::cout << "projection speed=" << p.speed << " angle=" << p.angle << " before=" << d.divergence_before
                << " after=" << d.divergence_after << " solid=" << d.solid_speed << " finite=" << d.finite << '\n';
      if (!d.finite || d.solid_speed != 0 || d.divergence_after > 0.02f * d.divergence_before + 0.0001f)
        return 2;
    }
    if (!sim.Reset(options.parameters))
      return 1;
  }
  if (!Render(device.Get(), runtime, options)) {
    std::cerr << "Wing fluid failed: " << sim.Error() << '\n';
    return 1;
  }
  return 0;
}
