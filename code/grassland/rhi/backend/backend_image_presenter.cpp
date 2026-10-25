#include "grassland/rhi/backend/backend_image_presenter.h"

#include <cmath>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {
namespace {
constexpr const char *shader = R"(
struct Settings { float white_scale; float3 padding; };
RHI_UNIFORM(0, 0, Settings, settings);
RHI_TEXTURE2D(0, 1, float4, source);
RHI_SAMPLER(0, 2, sourceSampler);
struct Varying { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Varying VSMain(uint id : SV_VertexID) {
  Varying o;
  o.uv = float2((id << 1) & 2, id & 2);
  o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
  return o;
}
#if RHI_PLATFORM_MACOS
static const float nits_per_unit = 100.0;
#elif RHI_PLATFORM_LINUX
static const float nits_per_unit = 203.0;
#else
static const float nits_per_unit = 80.0;
#endif
float3 Decode(float3 c) {
#if SOURCE_COLOR_SPACE == 0
  return select(c <= 0.04045, c / 12.92, pow((max(c, 0) + 0.055) / 1.055, 2.4));
#elif SOURCE_COLOR_SPACE == 1
  return c / settings.white_scale;
#elif SOURCE_COLOR_SPACE == 2
  float3 p = pow(saturate(c), 32.0 / 2523.0);
  float3 linear2020 = pow(max(p - 3424.0 / 4096.0, 0) /
                         max(2413.0 / 128.0 - (2392.0 / 128.0) * p, 1e-6), 16384.0 / 2610.0);
  const float3x3 to709 = float3x3(1.660491, -0.587641, -0.072850,
                                -0.124550, 1.132900, -0.008349,
                                -0.018151, -0.100579, 1.118730);
  return mul(to709, linear2020) * (10000.0 / (settings.white_scale * nits_per_unit));
#else
  return c;
#endif
}
float3 Encode(float3 c) {
#if TARGET_COLOR_SPACE == 0
  c = saturate(c);
  return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
#elif TARGET_COLOR_SPACE == 1
  return c * settings.white_scale;
#elif TARGET_COLOR_SPACE == 2
  const float3x3 to2020 = float3x3(0.627404, 0.329283, 0.043313,
                                 0.069097, 0.919540, 0.011362,
                                 0.016391, 0.088013, 0.895595);
  float3 normalized = saturate(mul(to2020, c) * (settings.white_scale * nits_per_unit / 10000.0));
  float3 p = pow(normalized, 2610.0 / 16384.0);
  return pow((3424.0 / 4096.0 + (2413.0 / 128.0) * p) / (1 + (2392.0 / 128.0) * p), 2523.0 / 32.0);
#else
  return c;
#endif
}
float4 PSMain(Varying input) : SV_Target0 {
  float4 c = source.Sample(sourceSampler, input.uv);
#if SOURCE_COLOR_SPACE != TARGET_COLOR_SPACE
  c.rgb = Encode(Decode(c.rgb));
#endif
  return c;
}
)";
}

ImagePresenter::ImagePresenter(Device *device) : device_(device) {
}

bool ImagePresenter::Record(ICommandContext *commands,
                            IImage *source,
                            IImage *target,
                            ColorSpace source_space,
                            ColorSpace target_space,
                            float white_scale) {
  auto valid_space = [](ColorSpace space) {
    return space >= COLOR_SPACE_SRGB_NONLINEAR && space <= COLOR_SPACE_BT709_LINEAR;
  };
  auto valid_format = [](ImageFormat format) {
    return IsColorFormat(format) && !IsSignedIntegerFormat(format) && !IsUnsignedIntegerFormat(format);
  };
  if (!commands || !source || !target || !valid_space(source_space) || !valid_space(target_space) ||
      !valid_format(source->GetDesc().format) || !valid_format(target->GetDesc().format) ||
      !std::isfinite(white_scale) || white_scale <= 0 || (source == target && source_space != target_space)) {
    ReportError(-1, "Image presentation requires distinct non-integer color images and valid color spaces.");
    return false;
  }
  if (source == target)
    return true;
  const auto extent = target->GetDesc();
  auto &program = programs_[{extent.format, source_space, target_space}];
  if (!program) {
    const auto code = "#define SOURCE_COLOR_SPACE " + std::to_string(int(source_space)) +
                      "\n#define TARGET_COLOR_SPACE " + std::to_string(int(target_space)) + "\n" + shader;
    ObjPtr<IShader> vs = device_->CreateShaderFromSource(code, "VSMain", SHADER_STAGE_VERTEX);
    ObjPtr<IShader> ps = device_->CreateShaderFromSource(code, "PSMain", SHADER_STAGE_FRAGMENT);
    if (!vs || !ps)
      return false;
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.color_formats = {extent.format};
    desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                     {1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                     {2, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
    program = device_->CreateProgram(desc);
    if (!program)
      return false;
  }
  if (!settings_)
    settings_ = device_->CreateBuffer({256});
  if (!sampler_)
    sampler_ =
        device_->CreateSampler({SAMPLER_FILTER_LINEAR, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_CLAMP_TO_EDGE});
  ObjPtr<IBindingGroup> group = device_->CreateBindingGroup(program->GetDesc().pipeline_layout.groups[0]);
  const float settings[] = {white_scale, 0, 0, 0};
  return settings_ && sampler_ && group && group->BindResource(0, settings_.Get()) && group->BindResource(1, source) &&
         group->BindResource(2, sampler_.Get()) &&
         commands->CmdUploadBuffer(settings_.Get(), 0, settings, sizeof(settings)) &&
         commands->CmdBindProgram(program.Get()) && commands->CmdBindGroup(0, group.Get()) &&
         commands->CmdBeginRendering({{{target, ATTACHMENT_LOAD_OP_DISCARD, ATTACHMENT_STORE_OP_STORE}}}) &&
         commands->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) &&
         commands->CmdSetScissor({0, 0, extent.width, extent.height}) && commands->CmdDraw(3) &&
         commands->CmdEndRendering();
}
}  // namespace grassland::rhi::backend
