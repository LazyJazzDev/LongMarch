#include "render.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <glm/ext.hpp>
#include <iostream>
#include <thread>

#include "imgui.h"

namespace wind3d {
namespace {
using namespace grassland::rhi;
#include "display.slang.h"

bool RenderFailure(int line) {
  std::cerr << "Wind3D renderer failed at render.cpp:" << line << "\n";
  return false;
}

Vec3 SampleVelocity(const Snapshot &snapshot, Vec3 p) {
  float h = 16.0f / snapshot.nx;
  float gx = (p.x + 6) / h - 0.5f, gy = (p.y + 4) / h - 0.5f, gz = (p.z + 6) / h - 0.5f;
  int x = int(floorf(gx)), y = int(floorf(gy)), z = int(floorf(gz));
  Vec3 velocity{};
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        int a = std::clamp(x + i, 0, snapshot.nx - 1), b = std::clamp(y + j, 0, snapshot.ny - 1),
            c = std::clamp(z + k, 0, snapshot.nz - 1);
        const auto &cell = snapshot.cells[(c * snapshot.ny + b) * snapshot.nx + a];
        float weight = (i ? gx - x : 1 - gx + x) * (j ? gy - y : 1 - gy + y) * (k ? gz - z : 1 - gz + z);
        velocity = Add(velocity, Scale({cell.u, cell.v, cell.w}, weight));
      }
  return velocity;
}

void Line(std::vector<Vertex> &lines, Vec3 a, Vec3 b, Vec3 color) {
  lines.push_back({a, {}, color});
  lines.push_back({b, {}, color});
}

void Arrow(std::vector<Vertex> &lines, Vec3 origin, Vec3 vector, Vec3 color) {
  float length = Length(vector);
  if (length < 1e-6f)
    return;
  Vec3 direction = Scale(vector, 1 / length);
  Vec3 perpendicular = Cross(direction, fabsf(direction.y) < 0.9f ? Vec3{0, 1, 0} : Vec3{0, 0, 1});
  perpendicular = Scale(perpendicular, 0.12f / Length(perpendicular));
  Vec3 end = Add(origin, vector), base = Add(end, Scale(direction, -0.28f));
  Line(lines, origin, end, color);
  Line(lines, end, Add(base, perpendicular), color);
  Line(lines, end, Add(base, Scale(perpendicular, -1)), color);
}

std::vector<Vertex> FlowLines(const Snapshot &snapshot, bool streamlines, bool loads) {
  std::vector<Vertex> lines;
  for (int x = -5; x <= 9; ++x)
    Line(lines, {float(x), -2, -5}, {float(x), -2, 5}, {0.055f, 0.08f, 0.12f});
  for (int z = -5; z <= 5; ++z)
    Line(lines, {-5, -2, float(z)}, {9, -2, float(z)}, {0.055f, 0.08f, 0.12f});
  Arrow(lines, {-4, -1.8f, -4.7f}, {1, 0, 0}, {1, 0.25f, 0.2f});
  Arrow(lines, {-4, -1.8f, -4.7f}, {0, 1, 0}, {0.2f, 1, 0.3f});
  Arrow(lines, {-4, -1.8f, -4.7f}, {0, 0, 1}, {0.2f, 0.5f, 1});
  if (streamlines) {
    for (int iz = -10; iz <= 10; ++iz)
      for (int iy = -1; iy <= 2; ++iy) {
        Vec3 p{-4.5f, iy * 0.32f + 0.05f, iz * 0.30f};
        for (int step = 0; step < 170; ++step) {
          Vec3 velocity = SampleVelocity(snapshot, p);
          float speed = Length(velocity);
          if (speed < 0.05f)
            break;
          Vec3 mid = Add(p, Scale(velocity, 0.04f / speed));
          Vec3 v2 = SampleVelocity(snapshot, mid);
          Vec3 next = Add(p, Scale(v2, 0.08f / std::max(Length(v2), 0.05f)));
          if (next.x > 9 || fabsf(next.y) > 3.7f || fabsf(next.z) > 5.7f ||
              AircraftDistance(next, snapshot.controls) < 0)
            break;
          float ratio = std::clamp(speed / snapshot.controls.speed * 0.6f, 0.0f, 1.0f);
          Vec3 color{0.05f + ratio * ratio * 0.8f, 0.25f + 0.6f * ratio, 0.9f - 0.75f * ratio};
          Line(lines, p, next, color);
          p = next;
        }
      }
  }
  if (loads) {
    Vec3 origin{0, 2, 0};
    Line(lines, {}, origin, {0.3f, 0.35f, 0.4f});
    Arrow(lines, origin, Scale(snapshot.force, 3 / (100 + Length(snapshot.force))), {0.1f, 1, 0.8f});
    Arrow(lines, origin, Scale(snapshot.moment, 3 / (100 + Length(snapshot.moment))), {1, 0.25f, 0.85f});
  }
  return lines;
}
}  // namespace

bool Render(IDevice *device,
            Simulation &simulation,
            Controls controls,
            int frames,
            bool headless,
            const std::string &capture,
            const ViewSettings &view) {
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IWindow> window;
  if (!headless) {
    window = device->CreateGLFWWindow({1440, 900, "Wind3D | asynchronous CUDA wind tunnel"});
    if (!window || !window->EnableImGui(true, LONGMARCH_ASSETS_DIR "fonts/simhei.ttf", 18))
      return RenderFailure(__LINE__);
  }
  ObjPtr<IBuffer> uniform = device->CreateBuffer({256});
  BindingLayoutDesc layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}};
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  if (!queue || !uniform || !group || !group->BindResource(0, uniform.Get()))
    return RenderFailure(__LINE__);
  std::string source(reinterpret_cast<const char *>(display_slang), display_slang_len);
  ObjPtr<IShader> vs = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> ps = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!vs || !ps)
    return RenderFailure(__LINE__);
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT3, 12}, {2, 0, INPUT_TYPE_FLOAT3, 24}};
  desc.pipeline_layout.groups = {layout};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = true;
  desc.depth_stencil.depth_write = true;
  ObjPtr<IProgram> aircraft_program = device->CreateProgram(desc);
  desc.topology = PRIMITIVE_TOPOLOGY_LINE_LIST;
  ObjPtr<IProgram> line_program = device->CreateProgram(desc);
  if (!aircraft_program || !line_program)
    return RenderFailure(__LINE__);
  ObjPtr<IImage> canvas, depth;
  ObjPtr<IBuffer> aircraft_buffer, line_buffer;
  ObjPtr<ISubmissionToken> last;

  // Ensure early failure also waits before destroying resources/window.
  struct WaitOnExit {
    ObjPtr<ISubmissionToken> &token;

    ~WaitOnExit() {
      if (token)
        token->WaitHost();
    }
  } wait{last};

  std::shared_ptr<const Snapshot> state;
  std::shared_ptr<const std::vector<Vertex>> mesh;
  size_t line_count = 0;
  bool paused = false, streamlines = view.flow, show_loads = view.loads, rebuild = true;
  float yaw = view.yaw, pitch = view.pitch, distance = view.distance, fps = 0, logical_width = 1440;
  auto previous = std::chrono::steady_clock::now(), beginning = previous;
  int rendered = 0, updates = 0;
  simulation.Configure(controls, paused);
  while (!frames || rendered < frames) {
    if (window && (!PollEvents() || window->ShouldClose()))
      break;
    if (!simulation.Error().empty()) {
      std::cerr << simulation.Error() << '\n';
      return RenderFailure(__LINE__);
    }
    if (last && !last->WaitHost())
      return RenderFailure(__LINE__);
    auto now = std::chrono::steady_clock::now();
    float seconds = std::chrono::duration<float>(now - previous).count();
    previous = now;
    fps = fps == 0 ? 1 / std::max(seconds, 0.0001f) : fps * 0.95f + 0.05f / std::max(seconds, 0.0001f);
    auto next = simulation.Latest();
    if (next && next != state) {
      state = next;
      rebuild = true;
      ++updates;
    }
    uint32_t width = 1440, height = 900;
    if (window) {
      if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
        return RenderFailure(__LINE__);
      ObjPtr<IImage> swap = window->GetNextImage();
      if (!swap) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      width = swap->GetDesc().width;
      height = swap->GetDesc().height;
      if (!window->UpdateImGui([&] {
            logical_width = ImGui::GetIO().DisplaySize.x;
            ImGui::SetNextWindowPos({logical_width - 350, 0});
            ImGui::SetNextWindowSize({350, ImGui::GetIO().DisplaySize.y});
            ImGui::Begin("Wind3D", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::Text("Frontend %.1f fps", fps);
            if (state) {
              ImGui::Text("CUDA step %.1f ms | #%llu", state->step_seconds * 1000,
                          static_cast<unsigned long long>(state->step));
              ImGui::Text("Simulation %.3f s | dt %.4f", state->time, state->dt);
              ImGui::Text("Grid %d | pressure %.2g (%d PCG)", state->solver_nx, state->pressure_residual,
                          state->pressure_cycles);
              if (!state->pressure_converged)
                ImGui::TextColored({1, 0.5f, 0.2f, 1}, "Pressure tolerance not met");
              ImGui::Text("Divergence RMS %.4g -> %.4g", state->divergence_before, state->divergence_after);
            } else
              ImGui::TextUnformatted("Building first flow snapshot...");
            ImGui::SeparatorText("Wind and controls");
            ImGui::SliderFloat("Wind m/s", &controls.speed, 1, 30);
            ImGui::SliderFloat("Incidence", &controls.angle, -20, 25, "%.1f deg");
            ImGui::SliderFloat("Flaps", &controls.flap, 0, 40, "%.1f deg");
            ImGui::SliderFloat("Ailerons", &controls.aileron, -25, 25, "%.1f deg");
            ImGui::SliderFloat("Elevator", &controls.elevator, -25, 25, "%.1f deg");
            ImGui::SliderFloat("Rudder", &controls.rudder, -25, 25, "%.1f deg");
            ImGui::Checkbox("Pause solver", &paused);
            ImGui::SameLine();
            if (ImGui::Button("Reset"))
              simulation.Reset();
            ImGui::TextWrapped(
                "Controls apply at the next solver step. Model and flow display the same completed state.");
            ImGui::SeparatorText("View");
            rebuild |= ImGui::Checkbox("3D streamlines", &streamlines);
            rebuild |= ImGui::Checkbox("Force and moment vectors", &show_loads);
            ImGui::SliderFloat("Orbit", &yaw, -3.14f, 3.14f);
            ImGui::SliderFloat("Elevation", &pitch, -1.55f, 1.55f);
            ImGui::SliderFloat("Distance", &distance, 7, 25);
            ImGui::TextWrapped("Drag outside panel to orbit; wheel to zoom. Flow: blue = slow, yellow = fast.");
            ImGui::SeparatorText("Integrated aircraft loads");
            if (state) {
              ImGui::TextColored({0.1f, 1, 0.8f, 1}, "Force [N]");
              ImGui::Text("Fx % .3f\nFy % .3f\nFz % .3f", state->force.x, state->force.y, state->force.z);
              ImGui::TextColored({1, 0.25f, 0.85f, 1}, "Moment [N m], about body origin");
              ImGui::Text("Mx % .3f\nMy % .3f\nMz % .3f", state->moment.x, state->moment.y, state->moment.z);
            }
            ImGui::TextWrapped(
                "+x downstream, +y up, +z starboard. Moment follows right-hand rule. Arrows share an elevated origin with compressed length.");
            ImGui::TextWrapped(
                "Incompressible, laminar, voxel boundary. Qualitative demonstration; no turbulence or aerodynamic certification.");
            ImGui::End();
            auto &io = ImGui::GetIO();
            if (!io.WantCaptureMouse) {
              if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                yaw -= io.MouseDelta.x * 0.006f;
                pitch = std::clamp(pitch + io.MouseDelta.y * 0.006f, -1.55f, 1.55f);
              }
              distance = std::clamp(distance - io.MouseWheel, 7.0f, 25.0f);
            }
          }))
        return RenderFailure(__LINE__);
      simulation.Configure(controls, paused);
    }
    if (!canvas || canvas->GetDesc().width != width || canvas->GetDesc().height != height) {
      canvas = device->CreateImage({width, height, IMAGE_FORMAT_R8G8B8A8_UNORM});
      depth = device->CreateImage({width, height, IMAGE_FORMAT_D32_SFLOAT});
      if (!canvas || !depth)
        return RenderFailure(__LINE__);
    }
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands)
      return RenderFailure(__LINE__);
    if (state && mesh != state->aircraft) {
      mesh = state->aircraft;
      aircraft_buffer = device->CreateBuffer({mesh->size() * sizeof(Vertex)});
      if (!aircraft_buffer ||
          !commands->CmdUploadBuffer(aircraft_buffer.Get(), 0, mesh->data(), mesh->size() * sizeof(Vertex)))
        return RenderFailure(__LINE__);
    }
    if (state && rebuild) {
      auto lines = FlowLines(*state, streamlines, show_loads);
      line_count = lines.size();
      line_buffer = device->CreateBuffer({line_count * sizeof(Vertex)});
      if (!line_buffer || !commands->CmdUploadBuffer(line_buffer.Get(), 0, lines.data(), line_count * sizeof(Vertex)))
        return RenderFailure(__LINE__);
      rebuild = false;
    }
    float viewport_width =
        headless ? float(width) : std::max(1.0f, float(width) * (logical_width - 350) / logical_width);
    glm::vec3 eye{distance * cosf(pitch) * sinf(yaw), distance * sinf(pitch), distance * cosf(pitch) * cosf(yaw)};
    auto transform = glm::perspectiveZO(glm::radians(45.0f), viewport_width / height, 0.1f, 80.0f) *
                     glm::lookAt(eye, glm::vec3(0), glm::vec3(0, 1, 0));
    if (!commands->CmdUploadBuffer(uniform.Get(), 0, &transform, sizeof(transform)))
      return RenderFailure(__LINE__);
    RenderingInfo render;
    render.color_attachments = {
        {canvas.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.012f, 0.02f, 0.035f, 1}}};
    render.depth_stencil.image = depth.Get();
    render.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
    if (!commands->CmdBeginRendering(render) ||
        !commands->CmdSetViewport({0, 0, viewport_width, float(height), 0, 1}) ||
        !commands->CmdSetScissor({0, 0, width, height}))
      return RenderFailure(__LINE__);
    if (aircraft_buffer &&
        (!commands->CmdBindProgram(aircraft_program.Get()) || !commands->CmdBindGroup(0, group.Get()) ||
         !commands->CmdBindVertexBuffer(0, aircraft_buffer.Get()) || !commands->CmdDraw(uint32_t(mesh->size()))))
      return RenderFailure(__LINE__);
    if (line_buffer &&
        (!commands->CmdBindProgram(line_program.Get()) || !commands->CmdBindGroup(0, group.Get()) ||
         !commands->CmdBindVertexBuffer(0, line_buffer.Get()) || !commands->CmdDraw(uint32_t(line_count))))
      return RenderFailure(__LINE__);
    if (!commands->CmdEndRendering())
      return RenderFailure(__LINE__);
    if (window && (!commands->CmdDrawImGui(window.Get(), canvas.Get()) ||
                   !commands->CmdPresentImage(window.Get(), canvas.Get(), COLOR_SPACE_BT709_LINEAR)))
      return RenderFailure(__LINE__);
    ObjPtr<IDownloadToken> download;
    bool final_frame = frames && rendered + 1 == frames;
    if (final_frame && !capture.empty()) {
      download = commands->CmdDownloadImage(canvas.Get());
      if (!download)
        return RenderFailure(__LINE__);
    }
    last = commands->Submit();
    if (!last)
      return RenderFailure(__LINE__);
    if (download) {
      std::vector<unsigned char> pixels(size_t(width) * height * 4);
      if (!download->WaitHost() || !download->CopyTo(pixels.data(), pixels.size()))
        return RenderFailure(__LINE__);
      std::ofstream output(capture, std::ios::binary);
      output << "P6\n" << width << ' ' << height << "\n255\n";
      for (size_t i = 0; i < pixels.size(); i += 4)
        output.write(reinterpret_cast<const char *>(pixels.data() + i), 3);
      if (!output)
        return RenderFailure(__LINE__);
    }
    ++rendered;
    if (headless)
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
  }
  float duration = std::chrono::duration<float>(std::chrono::steady_clock::now() - beginning).count();
  std::cout << "frames=" << rendered << " snapshots=" << updates << " wall_seconds=" << duration
            << " fps=" << rendered / duration << '\n';
  return !last || last->WaitHost();
}
}  // namespace wind3d
