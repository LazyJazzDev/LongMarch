#include "DemoSession.h"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <random>
#include <stdexcept>

#include "demo/nbody_cs/params.h"
#include "grassland/graphics/backend/metal/metal_command_context.h"
#include "grassland/graphics/shader_cache.h"
namespace {
#include "demo_shaders.inl"
struct ColorVertex {
  glm::vec3 position, color;
};
struct BlendVertex {
  glm::vec3 position;
  glm::vec4 color;
};
struct TextureVertex {
  glm::vec3 position;
  glm::vec2 uv;
};
struct CubeUniform {
  glm::mat4 model, view, projection;
};
struct ParticleUniform {
  glm::mat4 world_to_screen, camera_to_world;
  float size;
  int hdr;
};
struct ParticleSettings {
  int count;
  float delta_time, gravity;
};
}  // namespace
using namespace grassland;
using namespace grassland::graphics;
const std::vector<std::string> &DemoSession::Names() {
  static const std::vector<std::string> names{"graphics_hello_triangle",   "graphics_hello_texture",
                                              "graphics_hello_blend",      "graphics_hello_resize",
                                              "graphics_hello_sdr_sample", "nbody_cs"};
  return names;
}
DemoSession::DemoSession(const std::filesystem::path &resources, const std::string &demo, bool prepare) : demo_(demo) {
  if (std::find(Names().begin(), Names().end(), demo) == Names().end())
    throw std::invalid_argument("Unknown graphics demo");
  ConfigureShaderCache({resources / "shaders", !prepare, true});
  if (CreateCore(BACKEND_API_METAL, Core::Settings{1, false}, &core_) ||
      core_->InitializeLogicalDeviceAutoSelect(false))
    throw std::runtime_error("Cannot initialize Metal");
  core_->CreateImage(width_, height_, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_);
  if (demo_ == "nbody_cs")
    InitializeNBody();
  else
    InitializeRaster();
}
DemoSession::~DemoSession() {
  try {
    core_->WaitGPU();
  } catch (...) {
  }
}
std::unique_ptr<Buffer> DemoSession::Buffer(const void *data, size_t size) {
  std::unique_ptr<graphics::Buffer> result;
  core_->CreateBuffer(size, BUFFER_TYPE_STATIC, &result);
  if (data)
    result->UploadData(data, size);
  return result;
}
void DemoSession::InitializeRaster() {
  auto vfs = GetShaderVirtualFileSystem();
  auto shader = demo_ + "/shaders/shader.hlsl";
  core_->CreateShader(vfs, shader, "VSMain", "vs_6_0", &vertex_);
  core_->CreateShader(vfs, shader, "PSMain", "ps_6_0", &fragment_);
  bool texture = demo_ == "graphics_hello_texture", cube = demo_ == "graphics_hello_resize";
  bool blend = demo_ == "graphics_hello_blend", sdr = demo_ == "graphics_hello_sdr_sample";
  if (texture || cube)
    core_->CreateImage(width_, height_, IMAGE_FORMAT_D32_SFLOAT, &depth_);
  core_->CreateProgram({color_->Format()}, depth_ ? depth_->Format() : IMAGE_FORMAT_UNDEFINED, &program_);
  const uint32_t triangle_indices[] = {0, 1, 2};
  if (!sdr)
    indices_ = Buffer(triangle_indices, sizeof(triangle_indices));
  if (texture) {
    const TextureVertex vertices[] = {
        {{0, 0.5f, 0}, {0.5f, 0}}, {{-0.5f, -0.5f, 0}, {0, 1}}, {{0.5f, -0.5f, 0}, {1, 1}}};
    vertices_ = Buffer(vertices, sizeof(vertices));
    core_->CreateImage(256, 256, IMAGE_FORMAT_R8G8B8A8_UNORM, &texture_);
    std::vector<uint32_t> pixels(256 * 256);
    for (uint32_t y = 0; y < 256; ++y)
      for (uint32_t x = 0; x < 256; ++x) {
        uint32_t p = x ^ y;
        pixels[y * 256 + x] = p | (p << 8) | (p << 16) | 0xff000000;
      }
    texture_->UploadData(pixels.data());
    core_->CreateSampler(FILTER_MODE_LINEAR, &sampler_);
    program_->AddInputBinding(sizeof(TextureVertex), false);
    program_->AddInputAttribute(0, INPUT_TYPE_FLOAT3, 0);
    program_->AddInputAttribute(0, INPUT_TYPE_FLOAT2, 12);
    program_->AddResourceBinding(RESOURCE_TYPE_IMAGE, 1);
    program_->AddResourceBinding(RESOURCE_TYPE_SAMPLER, 1);
  } else if (blend) {
    const BlendVertex vertices[] = {{{0, .5f, 0}, {0, 1, 1, .5f}},    {{-.5f, -.5f, 0}, {1, 1, 0, .5f}},
                                    {{.5f, -.5f, 0}, {1, 0, 1, .5f}}, {{0, .25f, 0}, {1, 0, 0, 1}},
                                    {{-.3f, -.35f, 0}, {0, 0, 1, 1}}, {{.3f, -.35f, 0}, {0, 1, 0, 1}}};
    vertices_ = Buffer(vertices, sizeof(vertices));
    program_->AddInputBinding(sizeof(BlendVertex), false);
    program_->AddInputAttribute(0, INPUT_TYPE_FLOAT3, 0);
    program_->AddInputAttribute(0, INPUT_TYPE_FLOAT4, 12);
    program_->SetBlendState(0, true);
  } else if (!sdr) {
    if (cube) {
      const ColorVertex vertices[] = {{{-1, -1, 1}, {0, 0, 0}}, {{1, -1, 1}, {1, 0, 0}},   {{-1, 1, 1}, {0, 1, 0}},
                                      {{1, 1, 1}, {1, 1, 0}},   {{-1, -1, -1}, {0, 0, 1}}, {{1, -1, -1}, {1, 0, 1}},
                                      {{-1, 1, -1}, {0, 1, 1}}, {{1, 1, -1}, {1, 1, 1}}};
      const uint32_t indices[] = {0, 1, 2, 2, 1, 3, 2, 3, 6, 6, 3, 7, 6, 7, 4, 4, 7, 5,
                                  4, 5, 0, 0, 5, 1, 1, 5, 3, 3, 5, 7, 0, 2, 4, 4, 2, 6};
      vertices_ = Buffer(vertices, sizeof(vertices));
      indices_ = Buffer(indices, sizeof(indices));
      index_count_ = 36;
      uniform_ = Buffer(nullptr, sizeof(CubeUniform));
      program_->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
      program_->SetCullMode(CULL_MODE_NONE);
    } else {
      const ColorVertex vertices[] = {
          {{0, .5f, 0}, {1, 0, 0}}, {{-.5f, -.5f, 0}, {0, 0, 1}}, {{.5f, -.5f, 0}, {0, 1, 0}}};
      vertices_ = Buffer(vertices, sizeof(vertices));
    }
    program_->AddInputBinding(sizeof(ColorVertex), false);
    program_->AddInputAttribute(0, INPUT_TYPE_FLOAT3, 0);
    program_->AddInputAttribute(0, INPUT_TYPE_FLOAT3, 12);
  }
  program_->BindShader(vertex_.get(), SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_.get(), SHADER_TYPE_PIXEL);
  program_->Finalize();
}
void DemoSession::InitializeNBody() {
  auto vfs = GetShaderVirtualFileSystem();
  core_->CreateShader(vfs, "nbody_cs/shaders/nbody.hlsl", "CSMain", "cs_6_0", &compute_);
  core_->CreateComputeProgram(compute_.get(), &compute_program_);
  compute_program_->AddResourceBinding(RESOURCE_TYPE_STORAGE_BUFFER, 1);
  compute_program_->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program_->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  compute_program_->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  compute_program_->Finalize();
  core_->CreateShader(vfs, "nbody_cs/shaders/particle.hlsl", "VSMain", "vs_6_0", &vertex_);
  core_->CreateShader(vfs, "nbody_cs/shaders/particle.hlsl", "PSMain", "ps_6_0", &fragment_);
  core_->CreateProgram({color_->Format()}, IMAGE_FORMAT_UNDEFINED, &program_);
  program_->SetBlendState(0, BlendState(BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_ADD, BLEND_FACTOR_ONE,
                                        BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, BLEND_OP_ADD));
  program_->AddInputBinding(sizeof(glm::vec3), true);
  program_->AddInputAttribute(0, INPUT_TYPE_FLOAT3, 0);
  program_->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->BindShader(vertex_.get(), SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_.get(), SHADER_TYPE_PIXEL);
  program_->Finalize();
  uniform_ = Buffer(nullptr, sizeof(ParticleUniform));
  settings_ = Buffer(nullptr, sizeof(ParticleSettings));
  ResetParticles();
}
void DemoSession::ResetParticles() {
  // Same deterministic ten-galaxy initial distribution as the desktop demo (seed 1).
  std::mt19937 random(1);
  auto scalar = [&]() { return std::uniform_real_distribution<float>()(random); };
  auto sphere = [&]() {
    float z = scalar() * 2 - 1, r = std::sqrt(1 - z * z), angle = scalar() * glm::pi<float>() * 2;
    return glm::vec3{r * std::sin(angle), r * std::cos(angle), z} * std::pow(scalar(), 1.0f / 3.0f);
  };
  std::vector<glm::vec3> origins(galaxies_), initial_velocities(galaxies_), positions(particles_),
      velocities(particles_);
  glm::vec3 average_pos{0}, average_vel{0};
  for (int i = 0; i < galaxies_; ++i) {
    origins[i] = sphere() * INITIAL_RADIUS * 2.0f;
    initial_velocities[i] = sphere() * INITIAL_RADIUS * .1f;
    average_pos += origins[i];
    average_vel += initial_velocities[i];
  }
  for (int i = 0; i < galaxies_; ++i) {
    origins[i] -= average_pos / float(galaxies_);
    initial_velocities[i] -= average_vel / float(galaxies_);
  }
  for (int i = 0; i < particles_; ++i) {
    int galaxy = std::uniform_int_distribution<int>(0, galaxies_ - 1)(random);
    positions[i] = sphere() * INITIAL_RADIUS * .2f * std::pow(10.0f / galaxies_, 1.0f / 3.0f) + origins[galaxy];
    velocities[i] = sphere() * INITIAL_SPEED + initial_velocities[galaxy];
  }
  positions_ = Buffer(positions.data(), positions.size() * sizeof(glm::vec3));
  velocities_ = Buffer(velocities.data(), velocities.size() * sizeof(glm::vec3));
  next_positions_ = Buffer(nullptr, positions.size() * sizeof(glm::vec3));
}
void DemoSession::Configure(int particles, int galaxies, float dt, bool simulate, float yaw, float pitch, int reset) {
  if (particles < 128 || particles > 65536 || particles % 128 || galaxies < 1 || galaxies > 20 || !std::isfinite(dt) ||
      dt < .001f || dt > .1f || !std::isfinite(yaw) || !std::isfinite(pitch))
    throw std::invalid_argument("Invalid simulation settings");
  bool changed = particles != particles_ || galaxies != galaxies_ || reset != reset_;
  particles_ = particles;
  galaxies_ = galaxies;
  delta_time_ = dt;
  simulate_ = simulate;
  yaw_ = yaw;
  pitch_ = pitch;
  reset_ = reset;
  if (changed && demo_ == "nbody_cs") {
    core_->WaitGPU();
    ResetParticles();
  }
}
void DemoSession::Resize(int width, int height) {
  if (width == width_ && height == height_)
    return;
  if (width < 1 || height < 1 || width > 8192 || height > 8192)
    throw std::invalid_argument("Invalid demo resolution");
  core_->WaitGPU();
  width_ = width;
  height_ = height;
  core_->CreateImage(width, height, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_);
  if (depth_)
    core_->CreateImage(width, height, IMAGE_FORMAT_D32_SFLOAT, &depth_);
}
void DemoSession::Render() {
  bool nbody = demo_ == "nbody_cs";
  if (nbody) {
    auto rotation =
        glm::rotate(glm::mat4(1), yaw_, glm::vec3(0, 1, 0)) * glm::rotate(glm::mat4(1), pitch_, glm::vec3(1, 0, 0));
    auto view = glm::lookAt(glm::vec3(10, 20, 30), glm::vec3(0), glm::vec3(0, 1, 0)) * rotation;
    ParticleUniform ubo{glm::perspective(glm::radians(60.f), float(width_) / height_, .1f, 100.f) * view,
                        glm::inverse(view), PARTICLE_SIZE, 0};
    // Keep total mass constant when selecting fewer particles on a mobile device.
    ParticleSettings settings{particles_, delta_time_, 100.0f / particles_};
    uniform_->UploadData(&ubo, sizeof(ubo));
    settings_->UploadData(&settings, sizeof(settings));
  } else if (demo_ == "graphics_hello_resize") {
    if (simulate_)
      theta_ += glm::radians(1.f);
    CubeUniform ubo{glm::rotate(glm::mat4(1), theta_, glm::vec3(0, 1, 0)),
                    glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0)),
                    glm::perspectiveZO(glm::radians(45.f), float(width_) / height_, 3.5f, 6.5f)};
    uniform_->UploadData(&ubo, sizeof(ubo));
  }
  std::unique_ptr<CommandContext> ctx;
  core_->CreateCommandContext(&ctx);
  if (nbody && simulate_) {
    ctx->CmdBindComputeProgram(compute_program_.get());
    ctx->CmdBindResources(0, {positions_.get()}, BIND_POINT_COMPUTE);
    ctx->CmdBindResources(1, {velocities_.get()}, BIND_POINT_COMPUTE);
    ctx->CmdBindResources(2, {next_positions_.get()}, BIND_POINT_COMPUTE);
    ctx->CmdBindResources(3, {settings_.get()}, BIND_POINT_COMPUTE);
    ctx->CmdDispatch(particles_ / 128, 1, 1);
    ctx->CmdCopyBuffer(positions_.get(), next_positions_.get(), positions_->Size());
  }
  ctx->CmdClearImage(color_.get(), nbody ? ClearValue{{0, 0, 0, 1}} : ClearValue{{.6f, .7f, .8f, 1}});
  if (depth_)
    ctx->CmdClearImage(depth_.get(), {{1.0f}});
  ctx->CmdBeginRendering({color_.get()}, depth_.get());
  ctx->CmdBindProgram(program_.get());
  ctx->CmdSetViewport({0, 0, float(width_), float(height_), 0, 1});
  ctx->CmdSetScissor({0, 0, uint32_t(width_), uint32_t(height_)});
  ctx->CmdSetPrimitiveTopology(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  if (nbody) {
    ctx->CmdBindVertexBuffers(0, {positions_.get()}, {0});
    ctx->CmdBindResources(0, {uniform_.get()});
    ctx->CmdDraw(6, particles_, 0, 0);
  } else if (demo_ == "graphics_hello_sdr_sample") {
    ctx->CmdDraw(6, 1, 0, 0);
  } else {
    ctx->CmdBindVertexBuffers(0, {vertices_.get()}, {0});
    ctx->CmdBindIndexBuffer(indices_.get(), 0);
    if (texture_) {
      ctx->CmdBindResources(0, {texture_.get()});
      ctx->CmdBindResources(1, {sampler_.get()});
    }
    if (uniform_)
      ctx->CmdBindResources(0, {uniform_.get()});
    if (demo_ == "graphics_hello_blend")
      ctx->CmdDrawIndexed(3, 1, 0, 3, 0);
    ctx->CmdDrawIndexed(index_count_, 1, 0, 0, 0);
  }
  ctx->CmdEndRendering();
  core_->SubmitCommandContext(ctx.get());
  core_->WaitGPU();
  auto native = static_cast<backend::MetalCommandContext *>(ctx.get())->Handle();
  gpu_ms_ = std::max(0.0, (native->GPUEndTime() - native->GPUStartTime()) * 1000);
}
std::vector<glm::vec3> DemoSession::Positions() const {
  std::vector<glm::vec3> result(demo_ == "nbody_cs" ? particles_ : 0);
  if (!result.empty())
    positions_->DownloadData(result.data(), result.size() * sizeof(glm::vec3));
  return result;
}
