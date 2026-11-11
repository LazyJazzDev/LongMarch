#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../rhix_hello/save_image.h"
#include "grassland/rhi/rhix/rhix.h"

namespace {
using namespace grassland::rhi;
namespace rx = grassland::rhix;
using Clock = std::chrono::steady_clock;

struct Options {
  BackendAPI api{BACKEND_API_VULKAN};
  uint32_t grid{64}, width{1280}, height{720}, batch{256}, iterations{1}, frames{300}, warmup{30};
  std::string capture;
};

bool Parse(int argc, char **argv, Options &o) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 == argc)
      return false;
    const std::string value = argv[++i];
    if (key == "--backend") {
      if (value == "vulkan")
        o.api = BACKEND_API_VULKAN;
      else if (value == "d3d12")
        o.api = BACKEND_API_D3D12;
      else
        return false;
    } else if (key == "--capture") {
      o.capture = value;
    } else {
      uint32_t *target = key == "--grid"         ? &o.grid
                         : key == "--width"      ? &o.width
                         : key == "--height"     ? &o.height
                         : key == "--batch"      ? &o.batch
                         : key == "--iterations" ? &o.iterations
                         : key == "--frames"     ? &o.frames
                         : key == "--warmup"     ? &o.warmup
                                                 : nullptr;
      if (!target)
        return false;
      auto parsed = std::from_chars(value.data(), value.data() + value.size(), *target);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return false;
    }
  }
  return o.grid && o.grid <= 128 && o.width && o.width <= 7680 && o.height && o.height <= 4320 && o.batch &&
         o.batch <= 16384 && o.iterations && o.iterations <= 2048 && o.frames && o.frames <= 100000 &&
         o.warmup <= 10000;
}

const char *shader = R"(
struct ObjectData { float4 rect; float4 tint; float4 motion; };
RHI_UNIFORM(0, 0, ObjectData, object);
RHI_TEXTURE2D(1, 0, float4, albedo);
RHI_SAMPLER(1, 1, linearSampler);
struct Vertex { RHI_INPUT(0, float2, position); RHI_INPUT(1, float2, uv); };
struct Varying { float4 position : SV_Position; float2 uv : TEXCOORD0; float4 tint : COLOR0; };
Varying VSMain(Vertex v) {
  Varying o;
  float s = sin(object.motion.x), c = cos(object.motion.x);
  float2 p = float2(c * v.position.x - s * v.position.y, s * v.position.x + c * v.position.y);
  o.position = float4(object.rect.xy + p * object.rect.zw, 0.5, 1);
  o.uv = v.uv; o.tint = object.tint;
  return o;
}
float4 PSMain(Varying v) : SV_Target0 {
  float3 color = albedo.Sample(linearSampler, v.uv).rgb * v.tint.rgb;
  // A dependent, bounded recurrence: work contributes to the final image.
  [loop] for (uint i = 0; i < SHADER_ITERATIONS; ++i) {
    color = frac(color.yzx * float3(1.031, 1.017, 1.023) + float3(0.071, 0.113, 0.097));
    color = color * (1.0 - color) * 3.75 + 0.015;
  }
  return float4(color * (0.85 + MATERIAL_VARIANT * 0.04), 1);
}
)";

struct Vertex {
  float position[2], uv[2];
};

struct alignas(256) ObjectData {
  float rect[4], tint[4], motion[4];
};

static_assert(sizeof(ObjectData) == 256);

int Run(const Options &o) {
  auto core = rx::CreateCore(o.api);
  if (!core)
    return 1;
  auto device = core.CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  auto queue = device.GetQueue(QUEUE_TYPE_MAIN);
  if (!queue)
    return 1;
  const uint32_t count = o.grid * o.grid;
  const uint32_t batches = (count + o.batch - 1) / o.batch;
  BindingLayoutDesc object_layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}};
  BindingLayoutDesc material_layout{
      {{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  auto vertex =
      device.CreateShaderFromSource(shader, "VSMain", SHADER_STAGE_VERTEX, {},
                                    {{"SHADER_ITERATIONS", std::to_string(o.iterations)}, {"MATERIAL_VARIANT", "0"}});
  if (!vertex)
    return 1;
  std::array<rx::Shader, 4> fragments;
  std::array<rx::Program, 4> programs;
  for (size_t i = 0; i < programs.size(); ++i) {
    fragments[i] = device.CreateShaderFromSource(
        shader, "PSMain", SHADER_STAGE_FRAGMENT, {},
        {{"SHADER_ITERATIONS", std::to_string(o.iterations)}, {"MATERIAL_VARIANT", std::to_string(i)}});
    if (!fragments[i])
      return 1;
    ProgramDesc p;
    p.vertex_shader = vertex.Get();
    p.fragment_shader = fragments[i].Get();
    p.vertex_bindings = {{sizeof(Vertex), false}};
    p.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT2, 0}, {1, 0, INPUT_TYPE_FLOAT2, 8}};
    p.color_formats = {IMAGE_FORMAT_R8G8B8A8_UNORM};
    p.pipeline_layout.groups = {object_layout, material_layout};
    p.rasterizer.cull_mode = CULL_MODE_NONE;
    programs[i] = device.CreateProgram(p);
    if (!programs[i])
      return 1;
  }
  std::array<rx::Buffer, 16> meshes;
  auto indices = device.CreateBuffer({6 * sizeof(uint32_t)});
  auto sampler = device.CreateSampler();
  if (!indices || !sampler)
    return 1;
  std::array<rx::Image, 8> textures;
  std::array<rx::BindingGroup, 8> materials;
  {
    auto upload = queue.CreateCommandContext();
    if (!upload)
      return 1;
    const uint32_t index_data[] = {0, 1, 2, 2, 1, 3};
    if (!upload.CmdUploadBuffer(indices, 0, index_data, sizeof(index_data)))
      return 1;
    for (size_t i = 0; i < meshes.size(); ++i) {
      float size = 0.66f + float(i) * 0.01f;
      const Vertex vertices[] = {
          {{-size, -size}, {0, 0}}, {{size, -size}, {1, 0}}, {{-size, size}, {0, 1}}, {{size, size}, {1, 1}}};
      meshes[i] = device.CreateBuffer({sizeof(vertices)});
      if (!meshes[i] || !upload.CmdUploadBuffer(meshes[i], 0, vertices, sizeof(vertices)))
        return 1;
    }
    for (size_t i = 0; i < textures.size(); ++i) {
      textures[i] = device.CreateImage({128, 128, IMAGE_FORMAT_R8G8B8A8_UNORM});
      materials[i] = device.CreateBindingGroup(material_layout);
      if (!textures[i] || !materials[i] || !materials[i].BindResource(0, textures[i]) ||
          !materials[i].BindResource(1, sampler))
        return 1;
      std::vector<uint32_t> pixels(128 * 128);
      for (uint32_t y = 0; y < 128; ++y)
        for (uint32_t x = 0; x < 128; ++x) {
          uint32_t v = ((x / 8 + y / 8) & 1) ? 180 : 60;
          pixels[y * 128 + x] = (v + i * 7) | ((v + i * 5) << 8) | ((v + i * 3) << 16) | 0xff000000;
        }
      if (!upload.CmdUploadImage(textures[i], pixels.data(), 128 * 4))
        return 1;
    }
    auto done = std::move(upload).Submit();
    if (!done || !done.WaitHost())
      return 1;
  }

  struct Frame {
    rx::Image image;
    rx::Buffer constants;
    std::vector<rx::BindingGroup> objects;
    rx::SubmissionToken done;
  };

  std::array<Frame, 3> slots;
  std::vector<ObjectData> objects(count);
  for (uint32_t i = 0; i < count; ++i) {
    auto &v = objects[i];
    v.rect[0] = (float(i % o.grid) + .5f) * 2 / o.grid - 1;
    v.rect[1] = (float(i / o.grid) + .5f) * 2 / o.grid - 1;
    v.rect[2] = v.rect[3] = 1.0f / o.grid;
    v.tint[0] = .4f + float((i * 17) % 61) * .01f;
    v.tint[1] = .4f + float((i * 23) % 61) * .01f;
    v.tint[2] = .4f + float((i * 31) % 61) * .01f;
    v.tint[3] = 1;
  }
  for (auto &slot : slots) {
    slot.image = device.CreateImage({o.width, o.height, IMAGE_FORMAT_R8G8B8A8_UNORM});
    slot.constants = device.CreateBuffer({uint64_t(count) * sizeof(ObjectData)});
    if (!slot.image || !slot.constants)
      return 1;
    slot.objects.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto group = device.CreateBindingGroup(object_layout);
      if (!group || !group.BindResource(0, rx::BufferView{slot.constants, uint64_t(i) * sizeof(ObjectData), 48}))
        return 1;
      slot.objects.push_back(std::move(group));
    }
  }
  auto drain = [&]() {
    for (auto &slot : slots)
      if (slot.done && !slot.done.WaitHost())
        return false;
    return true;
  };
  auto begin = Clock::now();
  double record_seconds = 0, wait_seconds = 0;
  for (uint32_t frame = 0; frame < o.warmup + o.frames; ++frame) {
    if (frame == o.warmup) {
      if (!drain())
        return 1;
      begin = Clock::now();
    }
    auto &slot = slots[frame % slots.size()];
    auto wait_begin = Clock::now();
    if (slot.done && !slot.done.WaitHost())
      return 1;
    slot.done = {};
    auto record_begin = Clock::now();
    for (uint32_t i = 0; i < count; ++i)
      objects[i].motion[0] = float(frame % 360) * .005f + float(i % 32) * .03f;
    for (uint32_t batch = 0; batch < batches; ++batch) {
      const uint32_t first = batch * o.batch, end = std::min(count, first + o.batch);
      auto commands = queue.CreateCommandContext();
      if (!commands ||
          !commands.CmdUploadBuffer(slot.constants, uint64_t(first) * sizeof(ObjectData), objects.data() + first,
                                    uint64_t(end - first) * sizeof(ObjectData)) ||
          !commands.CmdBindProgram(programs[batch % programs.size()]) ||
          !commands.CmdBeginRendering(slot.image, batch ? ATTACHMENT_LOAD_OP_LOAD : ATTACHMENT_LOAD_OP_CLEAR,
                                      ATTACHMENT_STORE_OP_STORE, {.02f, .03f, .04f, 1}) ||
          !commands.CmdSetViewport({0, 0, float(o.width), float(o.height), 0, 1}) ||
          !commands.CmdSetScissor({0, 0, o.width, o.height}))
        return 1;
      for (uint32_t i = first; i < end; ++i) {
        if ((i == first || i % 8 == 0) && !commands.CmdBindGroup(1, materials[(i / 8) % materials.size()]))
          return 1;
        if (!commands.CmdBindGroup(0, slot.objects[i]) || !commands.CmdBindVertexBuffer(0, meshes[i % meshes.size()]) ||
            !commands.CmdBindIndexBuffer(indices) || !commands.CmdDrawIndexed(6))
          return 1;
      }
      if (!commands.CmdEndRendering())
        return 1;
      slot.done = std::move(commands).Submit();
      if (!slot.done)
        return 1;
    }
    if (frame >= o.warmup) {
      record_seconds += std::chrono::duration<double>(Clock::now() - record_begin).count();
      wait_seconds += std::chrono::duration<double>(record_begin - wait_begin).count();
    }
  }
  if (!drain())
    return 1;
  const double wall = std::chrono::duration<double>(Clock::now() - begin).count();
  std::cout << fmt::format(
      "RESULT {{\"backend\":\"{}\",\"grid\":{},\"width\":{},\"height\":{},\"draws_per_frame\":{},\"submissions_per_frame\":{},\"iterations\":{},\"frames\":{},\"warmup\":{},\"in_flight\":3,\"seconds\":{:.9f},\"record_seconds\":{:.9f},\"wait_seconds\":{:.9f}}}\n",
      BackendAPIName(o.api), o.grid, o.width, o.height, count, batches, o.iterations, o.frames, o.warmup, wall,
      record_seconds, wait_seconds);
  if (!o.capture.empty()) {
    auto &slot = slots[(o.warmup + o.frames - 1) % slots.size()];
    auto commands = queue.CreateCommandContext();
    if (!commands)
      return 1;
    auto download = commands.CmdDownloadImage(slot.image);
    auto done = std::move(commands).Submit();
    if (!download || !done || !done.WaitHost() || !download.WaitHost())
      return 1;
    std::vector<uint8_t> pixels(download.GetSize());
    if (!download.CopyTo(pixels.data(), pixels.size()) ||
        !SaveImage(o.capture, "stress", pixels, slot.image.GetDesc(), COLOR_SPACE_SRGB_NONLINEAR, 1))
      return 1;
  }
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!Parse(argc, argv, options)) {
    std::cerr
        << "Usage: demo_rhi_stress --backend vulkan|d3d12 [--grid 64 --width 1280 --height 720 --batch 256 --iterations 1 --frames 300 --warmup 30 --capture DIR]\n";
    return 2;
  }
  return Run(options);
}
