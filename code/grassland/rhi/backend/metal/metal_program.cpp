#include "grassland/rhi/backend/metal/metal_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
MetalProgram::MetalProgram(MetalDevice *device, const ProgramDesc &desc) : Program(device, desc) {
}

MTL::RenderPipelineState *MetalProgram::NativePipeline() const {
  return pipeline_.get();
}

bool MetalProgram::Finalize() {
  if (finalized_)
    return true;
  try {
    if (!ValidateLayout() || desc_.vertex_bindings.size() > 31) {
      ReportError(-1, "Invalid Metal graphics program layout.");
      return false;
    }
    for (const auto &binding : desc_.vertex_bindings) {
      if (!binding.stride || binding.stride > 2048 || binding.stride % 4) {
        ReportError(-1, "Metal vertex stride must be a multiple of four in [4, 2048].");
        return false;
      }
    }
    for (const auto &attribute : desc_.vertex_attributes) {
      if (attribute.location >= 31 || attribute.offset % 4) {
        ReportError(-1, "Invalid Metal vertex attribute location or alignment.");
        return false;
      }
    }
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device = static_cast<MetalDevice *>(device_)->NativeDevice();
    NS::SharedPtr<MTL::Function> functions[2];
    const IShader *shaders[] = {desc_.vertex_shader, desc_.fragment_shader};
    auto prefix = ShaderMacroHeader(desc_, BACKEND_API_METAL);
    for (int i = 0; i < 2; ++i) {
      std::vector<uint8_t> code;
      if (!CompileSlang(shaders[i]->GetDesc(), prefix, BACKEND_API_METAL, code))
        return false;
      code.push_back(0);
      NS::Error *error = nullptr;
      auto library = NS::TransferPtr(device->newLibrary(
          NS::String::string(reinterpret_cast<const char *>(code.data()), NS::UTF8StringEncoding), nullptr, &error));
      if (!library) {
        ReportError(-1, error ? error->localizedDescription()->utf8String() : "Metal library compilation failed.");
        return false;
      }
      functions[i] = NS::TransferPtr(
          library->newFunction(NS::String::string(shaders[i]->GetDesc().entry_point.c_str(), NS::UTF8StringEncoding)));
      if (!functions[i]) {
        ReportError(-1, "Metal shader entry point not found.");
        return false;
      }
    }
    auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
    descriptor->setVertexFunction(functions[0].get());
    descriptor->setFragmentFunction(functions[1].get());
    descriptor->colorAttachments()->object(0)->setPixelFormat(NativeImageFormat(desc_.color_format));
    auto vertex = MTL::VertexDescriptor::vertexDescriptor();
    for (size_t i = 0; i < desc_.vertex_bindings.size(); ++i) {
      auto layout = vertex->layouts()->object(i);
      layout->setStride(desc_.vertex_bindings[i].stride);
      layout->setStepFunction(desc_.vertex_bindings[i].per_instance ? MTL::VertexStepFunctionPerInstance
                                                                    : MTL::VertexStepFunctionPerVertex);
      layout->setStepRate(1);
    }
    const MTL::VertexFormat formats[] = {MTL::VertexFormatFloat, MTL::VertexFormatFloat2, MTL::VertexFormatFloat3,
                                         MTL::VertexFormatFloat4};
    for (const auto &a : desc_.vertex_attributes) {
      auto attribute = vertex->attributes()->object(a.location);
      attribute->setFormat(formats[a.format]);
      attribute->setBufferIndex(a.binding);
      attribute->setOffset(a.offset);
    }
    descriptor->setVertexDescriptor(vertex);
    NS::Error *error = nullptr;
    pipeline_ = NS::TransferPtr(device->newRenderPipelineState(descriptor.get(), &error));
    if (!pipeline_) {
      ReportError(-1, error ? error->localizedDescription()->utf8String() : "Metal pipeline creation failed.");
      return false;
    }
    finalized_ = true;
    return true;
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Metal program allocation failed.");
    return false;
  }
}
}  // namespace grassland::rhi::backend::metal
