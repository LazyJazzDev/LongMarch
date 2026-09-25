#include "HdrPresentation.h"

#include "grassland/graphics/backend/vulkan/vulkan_shader.h"
#include "hdr_present.inc"

namespace longmarch::harmony {
using namespace grassland;

HdrPresentation::HdrPresentation(graphics::backend::VulkanCore *core) : core_(core) {
  graphics::CompiledShaderBlob blob;
  blob.entry_point = "main";
  blob.data.resize(sizeof(kHdrPresentShader));
  std::memcpy(blob.data.data(), kHdrPresentShader, sizeof(kHdrPresentShader));
  shader_ = std::make_unique<graphics::backend::VulkanShader>(core_, blob);
  core_->CreateComputeProgram(shader_.get(), &program_);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_IMAGE, 1);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program_->AddResourceBinding(graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->Finalize();
  core_->CreateBuffer(16, graphics::BUFFER_TYPE_DYNAMIC, &settings_);
}

graphics::Image *HdrPresentation::Convert(graphics::Image *source, bool hdr10, bool encoded_particles) {
  auto size = source->Extent();
  if (!output_ || output_->Extent().width != size.width || output_->Extent().height != size.height) {
    core_->WaitGPU();
    core_->CreateImage(size.width, size.height, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &output_);
  }
  const uint32_t settings[4] = {uint32_t(hdr10), uint32_t(encoded_particles), 0, 0};
  settings_->UploadData(settings, sizeof(settings));
  std::unique_ptr<graphics::CommandContext> commands;
  core_->CreateCommandContext(&commands);
  commands->CmdBindComputeProgram(program_.get());
  commands->CmdBindResources(0, {source}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(1, {output_.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdBindResources(2, {settings_.get()}, graphics::BIND_POINT_COMPUTE);
  commands->CmdDispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
  core_->SubmitCommandContext(commands.get());
  return output_.get();
}
}  // namespace longmarch::harmony
