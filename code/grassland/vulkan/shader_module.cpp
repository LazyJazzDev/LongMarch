#include "grassland/vulkan/shader_module.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include "grassland/graphics/graphics_util.h"

namespace CD::vulkan {

namespace {
EShLanguage VkShaderStageToEShLanguage(VkShaderStageFlagBits stage) {
  switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return EShLangVertex;
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return EShLangTessControl;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return EShLangTessEvaluation;
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      return EShLangGeometry;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return EShLangFragment;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return EShLangCompute;
      // Ray tracing shaders
    case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
      return EShLangRayGen;
    case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
      return EShLangAnyHit;
    case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
      return EShLangClosestHit;
    case VK_SHADER_STAGE_MISS_BIT_KHR:
      return EShLangMiss;
    case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
      return EShLangIntersect;
    case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
      return EShLangCallable;

    default:
      ThrowError("[Vulkan] invalid shader stage!");
  }
  return EShLangVertex;
}

TBuiltInResource InitResources() {
  TBuiltInResource Resources{};

  Resources.maxLights = 32;
  Resources.maxClipPlanes = 6;
  Resources.maxTextureUnits = 32;
  Resources.maxTextureCoords = 32;
  Resources.maxVertexAttribs = 64;
  Resources.maxVertexUniformComponents = 4096;
  Resources.maxVaryingFloats = 64;
  Resources.maxVertexTextureImageUnits = 32;
  Resources.maxCombinedTextureImageUnits = 80;
  Resources.maxTextureImageUnits = 32;
  Resources.maxFragmentUniformComponents = 4096;
  Resources.maxDrawBuffers = 32;
  Resources.maxVertexUniformVectors = 128;
  Resources.maxVaryingVectors = 8;
  Resources.maxFragmentUniformVectors = 16;
  Resources.maxVertexOutputVectors = 16;
  Resources.maxFragmentInputVectors = 15;
  Resources.minProgramTexelOffset = -8;
  Resources.maxProgramTexelOffset = 7;
  Resources.maxClipDistances = 8;
  Resources.maxComputeWorkGroupCountX = 65535;
  Resources.maxComputeWorkGroupCountY = 65535;
  Resources.maxComputeWorkGroupCountZ = 65535;
  Resources.maxComputeWorkGroupSizeX = 1024;
  Resources.maxComputeWorkGroupSizeY = 1024;
  Resources.maxComputeWorkGroupSizeZ = 64;
  Resources.maxComputeUniformComponents = 1024;
  Resources.maxComputeTextureImageUnits = 16;
  Resources.maxComputeImageUniforms = 8;
  Resources.maxComputeAtomicCounters = 8;
  Resources.maxComputeAtomicCounterBuffers = 1;
  Resources.maxVaryingComponents = 60;
  Resources.maxVertexOutputComponents = 64;
  Resources.maxGeometryInputComponents = 64;
  Resources.maxGeometryOutputComponents = 128;
  Resources.maxFragmentInputComponents = 128;
  Resources.maxImageUnits = 8;
  Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
  Resources.maxCombinedShaderOutputResources = 8;
  Resources.maxImageSamples = 0;
  Resources.maxVertexImageUniforms = 0;
  Resources.maxTessControlImageUniforms = 0;
  Resources.maxTessEvaluationImageUniforms = 0;
  Resources.maxGeometryImageUniforms = 0;
  Resources.maxFragmentImageUniforms = 8;
  Resources.maxCombinedImageUniforms = 8;
  Resources.maxGeometryTextureImageUnits = 16;
  Resources.maxGeometryOutputVertices = 256;
  Resources.maxGeometryTotalOutputComponents = 1024;
  Resources.maxGeometryUniformComponents = 1024;
  Resources.maxGeometryVaryingComponents = 64;
  Resources.maxTessControlInputComponents = 128;
  Resources.maxTessControlOutputComponents = 128;
  Resources.maxTessControlTextureImageUnits = 16;
  Resources.maxTessControlUniformComponents = 1024;
  Resources.maxTessControlTotalOutputComponents = 4096;
  Resources.maxTessEvaluationInputComponents = 128;
  Resources.maxTessEvaluationOutputComponents = 128;
  Resources.maxTessEvaluationTextureImageUnits = 16;
  Resources.maxTessEvaluationUniformComponents = 1024;
  Resources.maxTessPatchComponents = 120;
  Resources.maxPatchVertices = 32;
  Resources.maxTessGenLevel = 64;
  Resources.maxViewports = 16;
  Resources.maxVertexAtomicCounters = 0;
  Resources.maxTessControlAtomicCounters = 0;
  Resources.maxTessEvaluationAtomicCounters = 0;
  Resources.maxGeometryAtomicCounters = 0;
  Resources.maxFragmentAtomicCounters = 8;
  Resources.maxCombinedAtomicCounters = 8;
  Resources.maxAtomicCounterBindings = 1;
  Resources.maxVertexAtomicCounterBuffers = 0;
  Resources.maxTessControlAtomicCounterBuffers = 0;
  Resources.maxTessEvaluationAtomicCounterBuffers = 0;
  Resources.maxGeometryAtomicCounterBuffers = 0;
  Resources.maxFragmentAtomicCounterBuffers = 1;
  Resources.maxCombinedAtomicCounterBuffers = 1;
  Resources.maxAtomicCounterBufferSize = 16384;
  Resources.maxTransformFeedbackBuffers = 4;
  Resources.maxTransformFeedbackInterleavedComponents = 64;
  Resources.maxCullDistances = 8;
  Resources.maxCombinedClipAndCullDistances = 8;
  Resources.maxSamples = 4;
  Resources.maxMeshOutputVerticesNV = 256;
  Resources.maxMeshOutputPrimitivesNV = 512;
  Resources.maxMeshWorkGroupSizeX_NV = 32;
  Resources.maxMeshWorkGroupSizeY_NV = 1;
  Resources.maxMeshWorkGroupSizeZ_NV = 1;
  Resources.maxTaskWorkGroupSizeX_NV = 32;
  Resources.maxTaskWorkGroupSizeY_NV = 1;
  Resources.maxTaskWorkGroupSizeZ_NV = 1;
  Resources.maxMeshViewCountNV = 4;

  Resources.limits.nonInductiveForLoops = 1;
  Resources.limits.whileLoops = 1;
  Resources.limits.doWhileLoops = 1;
  Resources.limits.generalUniformIndexing = 1;
  Resources.limits.generalAttributeMatrixVectorIndexing = 1;
  Resources.limits.generalVaryingIndexing = 1;
  Resources.limits.generalSamplerIndexing = 1;
  Resources.limits.generalVariableIndexing = 1;
  Resources.limits.generalConstantMatrixVectorIndexing = 1;

  return Resources;
}

std::vector<uint32_t> ByteDataToDwordData(const std::vector<uint8_t> &byte_data) {
  std::vector<uint32_t> dword_data(byte_data.size() / sizeof(uint32_t));
  memcpy(dword_data.data(), byte_data.data(), byte_data.size());
  return dword_data;
}
}  // namespace

ShaderModule::ShaderModule(const struct Device *device, VkShaderModule shader_module, const std::string &entry_point)
    : device_(device), shader_module_(shader_module), entry_point_(entry_point) {
}

ShaderModule::~ShaderModule() {
  vkDestroyShaderModule(device_->Handle(), shader_module_, nullptr);
}

CompiledShaderBlob CompileGLSLToSPIRV(const std::string &glsl_code, VkShaderStageFlagBits shader_stage) {
  glslang::InitializeProcess();
  glslang::TShader shader(VkShaderStageToEShLanguage(shader_stage));
  shader.setEntryPoint("main");

  const char *shader_code = glsl_code.c_str();
  shader.setStrings(&shader_code, 1);

  // Use macro to downgrade vulkan version to 1.1 if on macOS
#ifdef __APPLE__
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
#else
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
#endif

  auto resources = InitResources();
  if (!shader.parse(&resources, 100, false, EShMsgDefault)) {
    SetErrorMessage("code:\n{}\n", glsl_code);
    // print error message
    SetErrorMessage("failed to parse shader!\nInfo: {}\n Debug Info: {}", shader.getInfoLog(),
                    shader.getInfoDebugLog());
  }
  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(EShMsgDefault)) {
    SetErrorMessage("failed to link shader!\nInfo: {}\n Debug Info: {}", program.getInfoLog(),
                    program.getInfoDebugLog());
  }
  std::vector<uint32_t> spirv_code;
  glslang::GlslangToSpv(*program.getIntermediate(VkShaderStageToEShLanguage(shader_stage)), spirv_code);
  glslang::FinalizeProcess();

  CompiledShaderBlob compiled_shader_blob;
  compiled_shader_blob.data.resize(spirv_code.size() * sizeof(uint32_t));
  std::memcpy(compiled_shader_blob.data.data(), spirv_code.data(), compiled_shader_blob.data.size());
  compiled_shader_blob.entry_point = "main";

  return compiled_shader_blob;
}

/*
 * #include <glslang/Public/ShaderLang.h>
#include <StandAlone/ResourceLimits.h>

#include <vector>

std::vector<uint32_t> compileGLSLToSPIRV(const std::string& glslCode,
EShLanguage shaderType) {
  // Initialize GLSLang
  glslang::InitializeProcess();

  // Create a new shader object
  glslang::TShader shader(shaderType);
  const char* shaderStrings[1] = { glslCode.c_str() };
  shader.setStrings(shaderStrings, 1);

  // Set some default options
  int clientInputSemanticsVersion = 100; // maps to #define VULKAN 100 in GLSL
  glslang::EShTargetClientVersion vulkanClientVersion =
glslang::EShTargetVulkan_1_0; glslang::EShTargetLanguageVersion targetVersion =
glslang::EShTargetSpv_1_0; shader.setEnvClient(glslang::EShClientVulkan,
clientInputSemanticsVersion); shader.setEnvTarget(glslang::EShTargetSpv,
targetVersion);

  // Invert y for Vulkan
  TBuiltInResource resources;
  resources = glslang::DefaultTBuiltInResource;
  EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
  if (!shader.parse(&resources, 100, false, messages)) {
    // Output error log
    printf("%s\n", shader.getInfoLog());
    printf("%s\n", shader.getInfoDebugLog());
  }

  // Perform final link of the shader
  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    // Output error log
    printf("%s\n", program.getInfoLog());
    printf("%s\n", program.getInfoDebugLog());
  }

  // Generate the SPIR-V binary
  std::vector<uint32_t> spirvBinary;
  glslang::GlslangToSpv(*program.getIntermediate(shaderType), spirvBinary);

  // Clean up
  glslang::FinalizeProcess();

  return spirvBinary;
}
 */

}  // namespace CD::vulkan
