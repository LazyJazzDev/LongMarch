#include "render.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <glm/ext.hpp>
#include <iostream>
#include <thread>

#include "imgui.h"
#include "load_charts.h"
#include "visualization.h"

namespace wind3d {
namespace {
using namespace grassland::rhi;
#include "display.slang.h"

bool RenderFailure(int line) {
  std::cerr << "Wind3D renderer failed at render.cpp:" << line << "\n";
  return false;
}

using visualization::FlowLines;

struct StreamSegment {
  Vec3 start, end, color, previous, next;
  float curvature_start, curvature_end;
};
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
  BindingLayoutDesc layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX | SHADER_VISIBILITY_FRAGMENT}}};
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
  desc.color_formats = {IMAGE_FORMAT_R8G8B8A8_UNORM, IMAGE_FORMAT_R32_SFLOAT};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = true;
  desc.depth_stencil.depth_write = true;
  ObjPtr<IProgram> aircraft_program = device->CreateProgram(desc);
  desc.topology = PRIMITIVE_TOPOLOGY_LINE_LIST;
  ObjPtr<IProgram> line_program = device->CreateProgram(desc);
  std::string stream_source = "#define WIND3D_STREAMLINES\n" + source;
  ObjPtr<IShader> stream_vs = device->CreateShaderFromSource(stream_source, "VSStream", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> stream_ps = device->CreateShaderFromSource(stream_source, "PSStream", SHADER_STAGE_FRAGMENT);
  if (!stream_vs || !stream_ps)
    return RenderFailure(__LINE__);
  ProgramDesc stream_desc = desc;
  stream_desc.vertex_shader = stream_vs.Get();
  stream_desc.fragment_shader = stream_ps.Get();
  stream_desc.topology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  stream_desc.vertex_bindings = {{sizeof(StreamSegment), true}};
  stream_desc.vertex_attributes.push_back({3, 0, INPUT_TYPE_FLOAT3, 36});
  stream_desc.vertex_attributes.push_back({4, 0, INPUT_TYPE_FLOAT3, 48});
  stream_desc.vertex_attributes.push_back({5, 0, INPUT_TYPE_FLOAT2, 60});
  stream_desc.color_formats = {IMAGE_FORMAT_R8G8B8A8_UNORM};
  stream_desc.depth_stencil.depth_write = false;
  stream_desc.blend_states = {BlendState(true)};
  ObjPtr<IProgram> stream_program = device->CreateProgram(stream_desc);
  if (!stream_program)
    return RenderFailure(__LINE__);
  BindingLayoutDesc volume_layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {1, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {2, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {3, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}};
  ObjPtr<IBindingGroup> volume_group = device->CreateBindingGroup(volume_layout);
  std::string volume_source = "#define WIND3D_VOLUME\n" + source;
  ObjPtr<IShader> volume_vs = device->CreateShaderFromSource(volume_source, "VSVolume", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> volume_ps = device->CreateShaderFromSource(volume_source, "PSVolume", SHADER_STAGE_FRAGMENT);
  ProgramDesc volume_desc;
  volume_desc.vertex_shader = volume_vs.Get();
  volume_desc.fragment_shader = volume_ps.Get();
  volume_desc.pipeline_layout.groups = {volume_layout};
  ObjPtr<IProgram> volume_program = device->CreateProgram(volume_desc);
  ObjPtr<IBuffer> volume_buffer = device->CreateBuffer({sizeof(float)});
  if (!volume_group || !volume_program || !volume_buffer || !volume_group->BindResource(0, uniform.Get()) ||
      !volume_group->BindResource(1, volume_buffer.Get()))
    return RenderFailure(__LINE__);
  if (!aircraft_program || !line_program)
    return RenderFailure(__LINE__);
  ObjPtr<IImage> canvas, scene, depth, surface_depth;
  ObjPtr<IBuffer> aircraft_buffer, line_buffer, stream_buffer;
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
  size_t stream_count = 0;
  int streamline_density = view.streamline_density;
  float streamline_width = view.streamline_width;
  float opaque_curvature = view.opaque_curvature;
  float transparent_curvature = view.transparent_curvature;
  bool paused = false, streamlines = view.flow, show_loads = view.loads, rebuild = true;
  bool show_volume = view.volume, colored_volume = false;
  float volume_density = 0.035f, volume_threshold = 6.0f, volume_step = 1.0f;
  std::shared_ptr<const Snapshot> volume_state;
  float yaw = view.yaw, pitch = view.pitch, distance = view.distance, fps = 0, logical_width = 1440;
  LoadHistory load_history;
  load_charts::Settings chart_settings;
  bool show_history = true;
  float logical_height = 900, chart_height = 0;
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
      if (window)
        load_history = simulation.RecentLoads();
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
            logical_height = ImGui::GetIO().DisplaySize.y;
            ImGui::SetNextWindowPos({logical_width - 410, 0});
            ImGui::SetNextWindowSize({410, ImGui::GetIO().DisplaySize.y});
            ImGui::Begin("Wind3D", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::Text("Frontend %.1f fps", fps);
            if (state) {
              ImGui::TextUnformatted(state->leapfrog_flow_maps ? "Leapfrog Flow Maps | 5 substeps/cycle"
                                                               : "MacCormack transport");
              ImGui::Text("CUDA step %.1f ms | #%llu", state->step_seconds * 1000,
                          static_cast<unsigned long long>(state->step));
              ImGui::Text("Simulation %.3f s | dt %.4f", state->time, state->dt);
              if (state->step_seconds > 0)
                ImGui::Text("Solver %.1f Hz | physical time %.2fx", 1 / state->step_seconds,
                            state->dt / state->step_seconds);
              ImGui::Text("Grid %d | pressure %.2g (%d PCG)", state->solver_nx, state->pressure_residual,
                          state->pressure_cycles);
              if (!state->pressure_converged)
                ImGui::TextColored({1, 0.5f, 0.2f, 1}, "Pressure tolerance not met");
              ImGui::Text("Divergence RMS %.4g -> %.4g", state->divergence_before, state->divergence_after);
            } else
              ImGui::TextUnformatted("Building first flow snapshot...");
            if (ImGui::CollapsingHeader("Wind and controls")) {
              ImGui::SliderFloat("Wind m/s", &controls.speed, 1, 30);
              ImGui::SliderFloat("Incidence", &controls.angle, -20, 25, "%.1f deg");
              ImGui::Checkbox("Link inner/outer flaps", &controls.wing_controls.linked_flaps);
              ImGui::SliderFloat(controls.wing_controls.linked_flaps ? "Flaps" : "Inner flap", &controls.flap, 0, 40,
                                 "%.1f deg");
              if (!controls.wing_controls.linked_flaps)
                ImGui::SliderFloat("Outer flap", &controls.wing_controls.outer_flap, 0, 40, "%.1f deg");
              ImGui::SliderFloat("Cruise camber", &controls.wing_controls.cruise_camber, -5, 5, "%.1f deg");
              ImGui::SliderFloat("Aileron droop", &controls.wing_controls.aileron_droop, 0, 15, "%.1f deg");
              ImGui::Checkbox("Link slats to flaps", &controls.wing_controls.linked_slats);
              if (!controls.wing_controls.linked_slats)
                ImGui::SliderFloat("Leading-edge slats", &controls.wing_controls.slat, 0, 30, "%.1f deg");
              ImGui::SliderFloat("Speed brakes", &controls.wing_controls.spoiler, 0, 60, "%.1f deg");
              ImGui::Checkbox("Link spoilers to ailerons", &controls.wing_controls.linked_spoiler_roll);
              if (!controls.wing_controls.linked_spoiler_roll)
                ImGui::SliderFloat("Spoiler roll", &controls.wing_controls.spoiler_roll, -30, 30, "%.1f deg");
              ImGui::SliderFloat("Ailerons", &controls.aileron, -25, 25, "%.1f deg");
              ImGui::Checkbox("Blended winglets", &controls.airframe.winglets);
              ImGui::SliderFloat("Stabilizer trim", &controls.airframe.tail_incidence, -10, 10, "%.2f deg");
              ImGui::SliderFloat("Elevator", &controls.elevator, -25, 25, "%.1f deg");
              ImGui::SliderFloat("Rudder", &controls.rudder, -25, 25, "%.1f deg");
            }
            ImGui::Checkbox("Pause solver", &paused);
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
              simulation.Reset();
              load_history = {};
            }
            ImGui::Checkbox("Load history plots", &show_history);
            if (show_volume && state && !state->vorticity.empty())
              ImGui::Text("Volume |curl u|: %d x %d x %d", state->solver_nx, state->solver_nx / 2,
                          state->solver_nx * 3 / 4);
            if (ImGui::CollapsingHeader("View")) {
              ImGui::Checkbox("Vorticity volume", &show_volume);
              ImGui::Checkbox("Color by vorticity", &colored_volume);
              ImGui::SliderFloat("Volume density", &volume_density, 0.01f, 0.6f, "%.3f");
              ImGui::SliderFloat("Threshold s^-1", &volume_threshold, 0, 60, "%.1f");
              ImGui::SliderFloat("Ray step / cell", &volume_step, 0.5f, 2, "%.2f");
              rebuild |= ImGui::Checkbox("3D streamlines", &streamlines);
              rebuild |= ImGui::SliderInt("Streamline density", &streamline_density, 1, 4);
              ImGui::SliderFloat("Streamline width (px)", &streamline_width, 0.75f, 4.0f);
              ImGui::SliderFloat("Transparent curvature (1/m)", &transparent_curvature, 0.0f, opaque_curvature - 0.001f,
                                 "%.3f", ImGuiSliderFlags_AlwaysClamp);
              ImGui::SliderFloat("Opaque curvature (1/m)", &opaque_curvature, transparent_curvature + 0.001f, 10.0f,
                                 "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
              ImGui::TextWrapped(
                  "Opacity remembers the largest upstream curvature. Below the lower threshold: transparent; at the upper threshold: opaque. Bends remain visible downstream.");
              rebuild |= ImGui::Checkbox("Force and moment vectors", &show_loads);
              ImGui::SliderFloat("Orbit", &yaw, -3.14f, 3.14f);
              ImGui::SliderFloat("Elevation", &pitch, -1.55f, 1.55f);
              ImGui::SliderFloat("Distance", &distance, 7, 25);
              ImGui::TextWrapped("Drag outside panel to orbit; wheel to zoom. Flow: blue = slow, yellow = fast.");
            }
            ImGui::SeparatorText("Integrated aircraft loads");
            if (state) {
              const auto &airframe = state->controls.airframe;
              ImGui::Text("Wing span %.2f m | camber %.1f%%", ProjectedWingSpan(state->controls),
                          100 * airframe.wing_camber);
              ImGui::TextColored({0.1f, 1, 0.8f, 1}, "World-axis force [N]");
              ImGui::Text("Fx % .3f\nFy % .3f\nFz % .3f", state->force.x, state->force.y, state->force.z);
              ImGui::Text("Magnitude %.3f N", Length(state->force));
              ImGui::TextColored({1, 0.25f, 0.85f, 1}, "World-axis moment [N m], about body origin");
              ImGui::Text("Mx % .3f\nMy % .3f\nMz % .3f", state->moment.x, state->moment.y, state->moment.z);
              ImGui::Text("Magnitude %.3f N m", Length(state->moment));
              Vec3 body_moment = RotateZ(state->moment, state->controls.angle);
              ImGui::Text("Body roll  % .3f\nBody pitch % .3f\nBody yaw   % .3f", body_moment.x, body_moment.z,
                          body_moment.y);
              ImGui::TextWrapped(
                  "Body moments [N m] use the same body origin and right-hand signs, not a specified center of mass.");
              ImGui::Text("Lift %.3f N | Drag %.3f N", state->force.y, state->force.x);
              if (std::abs(state->force.x) > 1e-4f)
                ImGui::Text("Instant L/D %.3f", state->force.y / state->force.x);
              else
                ImGui::TextUnformatted("Instant L/D undefined (near-zero drag)");
              if (state->time > 0) {
                double drag = state->impulse[0] / state->time, lift = state->impulse[1] / state->time;
                ImGui::Text("Mean lift %.3f N | drag %.3f N", lift, drag);
                if (std::abs(drag) > 1e-4)
                  ImGui::Text("Mean L/D %.3f", lift / drag);
                ImGui::TextWrapped(
                    "Means are time-weighted since reset, including startup. L/D is mean lift / mean drag.");
              }
            }
            ImGui::TextWrapped(
                "+x downstream, +y up, +z starboard. Moment follows right-hand rule. Arrows share an elevated origin with compressed length.");
            ImGui::TextWrapped(
                "Incompressible flow without a turbulence model. Absolute aircraft loads require aerodynamic reference validation.");
            ImGui::End();
            chart_height = load_charts::Draw(load_history, chart_settings, show_history, 410);
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
      scene = device->CreateImage({width, height, IMAGE_FORMAT_R8G8B8A8_UNORM});
      depth = device->CreateImage({width, height, IMAGE_FORMAT_D32_SFLOAT});
      surface_depth = device->CreateImage({width, height, IMAGE_FORMAT_R32_SFLOAT});
      if (!canvas || !scene || !depth || !surface_depth || !volume_group->BindResource(2, scene.Get()) ||
          !volume_group->BindResource(3, surface_depth.Get()))
        return RenderFailure(__LINE__);
    }
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands)
      return RenderFailure(__LINE__);
    if (state && state != volume_state && !state->vorticity.empty()) {
      volume_buffer = device->CreateBuffer({state->vorticity.size() * sizeof(float)});
      if (!volume_buffer ||
          !commands->CmdUploadBuffer(volume_buffer.Get(), 0, state->vorticity.data(),
                                     state->vorticity.size() * sizeof(float)) ||
          !volume_group->BindResource(1, volume_buffer.Get()))
        return RenderFailure(__LINE__);
      volume_state = state;
    }
    if (state && mesh != state->aircraft) {
      mesh = state->aircraft;
      aircraft_buffer = device->CreateBuffer({mesh->size() * sizeof(Vertex)});
      if (!aircraft_buffer ||
          !commands->CmdUploadBuffer(aircraft_buffer.Get(), 0, mesh->data(), mesh->size() * sizeof(Vertex)))
        return RenderFailure(__LINE__);
    }
    if (state && rebuild) {
      auto lines = FlowLines(*state, false, show_loads);
      line_count = lines.size();
      line_buffer = device->CreateBuffer({line_count * sizeof(Vertex)});
      if (!line_buffer || !commands->CmdUploadBuffer(line_buffer.Get(), 0, lines.data(), line_count * sizeof(Vertex)))
        return RenderFailure(__LINE__);
      auto flow = streamlines ? visualization::DomainFlowLines(*state, streamline_density) : std::vector<Vertex>{};
      std::vector<StreamSegment> segments;
      segments.reserve(flow.size() / 2);
      auto same_point = [](Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; };
      for (size_t i = 0; i + 1 < flow.size(); i += 2) {
        Vec3 start = flow[i].position, end = flow[i + 1].position;
        Vec3 previous = i >= 2 && same_point(flow[i - 1].position, start) ? flow[i - 2].position : start;
        Vec3 next = i + 2 < flow.size() && same_point(flow[i + 2].position, end) ? flow[i + 3].position : end;
        segments.push_back({start, end, flow[i].color, previous, next, flow[i].normal.x, flow[i + 1].normal.x});
      }
      stream_count = segments.size();
      if (stream_count) {
        stream_buffer = device->CreateBuffer({stream_count * sizeof(StreamSegment)});
        if (!stream_buffer ||
            !commands->CmdUploadBuffer(stream_buffer.Get(), 0, segments.data(), stream_count * sizeof(StreamSegment)))
          return RenderFailure(__LINE__);
      }
      rebuild = false;
    }
    float viewport_width =
        headless ? float(width) : std::max(1.0f, float(width) * (logical_width - 410) / logical_width);
    glm::vec3 eye{distance * cosf(pitch) * sinf(yaw), distance * sinf(pitch), distance * cosf(pitch) * cosf(yaw)};
    float viewport_height =
        headless ? float(height) : std::max(1.0f, float(height) * (logical_height - chart_height) / logical_height);
    auto transform = glm::perspectiveZO(glm::radians(45.0f), viewport_width / viewport_height, 0.1f, 80.0f) *
                     glm::lookAt(eye, glm::vec3(0), glm::vec3(0, 1, 0));

    struct Uniform {
      glm::mat4 transform, inverse;
      glm::vec4 eye;
      glm::uvec4 grid;
      glm::vec4 volume;
      glm::vec4 streamlines;
      glm::vec4 curvature;
    } globals{transform,
              glm::inverse(transform),
              glm::vec4(eye, 1),
              glm::uvec4(0),
              {show_volume ? volume_density : 0, volume_threshold, 0.1f, colored_volume ? 1.0f : 0.0f},
              {viewport_width, viewport_height, streamline_width, opaque_curvature},
              {transparent_curvature, 0, 0, 0}};

    if (volume_state && volume_state == state) {
      uint32_t nx = uint32_t(state->solver_nx);
      globals.grid = {nx, nx / 2, nx * 3 / 4, 0};
      globals.volume.z = volume_step * 16.0f / nx;
    }
    if (!commands->CmdUploadBuffer(uniform.Get(), 0, &globals, sizeof(globals)))
      return RenderFailure(__LINE__);
    RenderingInfo render;
    render.color_attachments = {
        {scene.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.012f, 0.02f, 0.035f, 1}},
        {surface_depth.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {1, 1, 1, 1}}};
    render.depth_stencil.image = depth.Get();
    render.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
    if (!commands->CmdBeginRendering(render) ||
        !commands->CmdSetViewport({0, 0, viewport_width, viewport_height, 0, 1}) ||
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
    RenderingInfo composite;
    composite.color_attachments = {{canvas.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}};
    if (!commands->CmdBeginRendering(composite) || !commands->CmdBindProgram(volume_program.Get()) ||
        !commands->CmdBindGroup(0, volume_group.Get()) || !commands->CmdDraw(3) || !commands->CmdEndRendering())
      return RenderFailure(__LINE__);
    if (stream_count) {
      RenderingInfo overlay;
      overlay.color_attachments = {{canvas.Get(), ATTACHMENT_LOAD_OP_LOAD, ATTACHMENT_STORE_OP_STORE}};
      overlay.depth_stencil.image = depth.Get();
      overlay.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_LOAD;
      if (!commands->CmdBeginRendering(overlay) || !commands->CmdBindProgram(stream_program.Get()) ||
          !commands->CmdSetViewport({0, 0, viewport_width, viewport_height, 0, 1}) ||
          !commands->CmdBindGroup(0, group.Get()) || !commands->CmdBindVertexBuffer(0, stream_buffer.Get()) ||
          !commands->CmdDraw(6, uint32_t(stream_count)) || !commands->CmdEndRendering())
        return RenderFailure(__LINE__);
    }
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
    else
      // The renderer otherwise spins above 1000 FPS and competes with CUDA for
      // the same GPU. Keep camera/UI updates at 60 Hz independently of solving.
      std::this_thread::sleep_until(now + std::chrono::microseconds(16667));
  }
  float duration = std::chrono::duration<float>(std::chrono::steady_clock::now() - beginning).count();
  std::cout << "frames=" << rendered << " snapshots=" << updates << " wall_seconds=" << duration
            << " fps=" << rendered / duration << '\n';
  return !last || last->WaitHost();
}
}  // namespace wind3d
