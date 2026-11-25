#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include "../../demo/rhi_hello/modules/pcf/scene.h"
#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#endif

namespace grassland::rhi {
// Keep native validation active for depth attachment -> sampled image transitions.
class RhiPCF : public testing::TestWithParam<BackendAPI> {
 protected:
  void SetUp() override {
    core_ = CreateCore(GetParam(), {true});
    ASSERT_TRUE(core_);
    device_ = core_->CreateDeviceByAutoSelection();
    ASSERT_TRUE(device_);
#if defined(LONGMARCH_D3D12_ENABLED)
    if (GetParam() == BACKEND_API_D3D12)
      ASSERT_TRUE(SUCCEEDED(QueryInterfacePtr<backend::d3d12::ID3D12Device>(device_.Get())
                                ->NativeDevice()
                                ->QueryInterface(IID_PPV_ARGS(&messages_))));
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (GetParam() == BACKEND_API_VULKAN) {
      instance_ = QueryInterfacePtr<backend::vulkan::IVulkanDevice>(device_.Get())->Instance();
      auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
      ASSERT_TRUE(create);
      VkDebugUtilsMessengerCreateInfoEXT desc{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      desc.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      desc.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
      desc.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                const VkDebugUtilsMessengerCallbackDataEXT *data, void *) -> VkBool32 {
        ADD_FAILURE() << data->pMessage;
        return VK_FALSE;
      };
      ASSERT_EQ(create(instance_, &desc, nullptr, &messenger_), VK_SUCCESS);
    }
#endif
  }

  void TearDown() override {
#if defined(LONGMARCH_VULKAN_ENABLED)
    if (messenger_) {
      auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
          vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
      destroy(instance_, messenger_, nullptr);
    }
#endif
#if defined(LONGMARCH_D3D12_ENABLED)
    if (messages_) {
      for (UINT64 i = 0; i < messages_->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T size = 0;
        messages_->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> bytes(size);
        auto message = reinterpret_cast<D3D12_MESSAGE *>(bytes.data());
        messages_->GetMessage(i, message, &size);
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
          ADD_FAILURE() << message->pDescription;
      }
    }
#endif
  }

  ObjPtr<ICore> core_;
  ObjPtr<IDevice> device_;
#if defined(LONGMARCH_D3D12_ENABLED)
  backend::d3d12::ComPtr<ID3D12InfoQueue> messages_;
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
  VkInstance instance_{};
  VkDebugUtilsMessengerEXT messenger_{};
#endif
};

TEST_P(RhiPCF, ComparisonOperatorsUseReferenceAgainstDepth) {
  ObjPtr<IImage> depth = device_->CreateImage({2, 2, IMAGE_FORMAT_D32_SFLOAT});
  ObjPtr<IBuffer> output = device_->CreateBuffer({3 * 4 * sizeof(float)});
  ASSERT_TRUE(depth && output);
  ObjPtr<IShader> shader = device_->CreateShaderFromSource(R"(
    RHI_TEXTURE2D(0,0,float,depth);
    RHI_COMPARISON_SAMPLER(0,1,comparison);
    RHI_SAMPLER(0,2,ordinary);
    RHI_RW_STRUCTURED_BUFFER(0,3,float4,results);
    RHI_COMPARISON_SAMPLER_ARRAY(0,4,comparisons);
    [numthreads(1,1,1)] void Main(uint3 id:SV_DispatchThreadID) {
      float reference = .25 * (id.x + 1);
      results[id.x] = float4(depth.SampleCmpLevelZero(comparison,float2(.5,.5),reference),
                             depth.SampleLevel(ordinary,float2(.5,.5),0),
                             depth.SampleCmpLevelZero(comparisons[0],float2(.5,.5),.75),
                             depth.SampleCmpLevelZero(comparisons[1],float2(.5,.5),.25));
    }
  )",
                                                           "Main", SHADER_STAGE_COMPUTE);
  ASSERT_TRUE(shader);
  BindingLayoutDesc layout{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                            {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_COMPUTE},
                            {2, BINDING_SAMPLER, 1, SHADER_VISIBILITY_COMPUTE},
                            {3, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 16},
                            {4, BINDING_SAMPLER, 2, SHADER_VISIBILITY_COMPUTE}}};
  ObjPtr<IComputeProgram> program = device_->CreateComputeProgram({shader.Get(), {{layout}}});
  ObjPtr<IBindingGroup> group = device_->CreateBindingGroup(layout);
  ObjPtr<ISampler> ordinary = device_->CreateSampler({SAMPLER_FILTER_NEAREST});
  ASSERT_TRUE(program && group && ordinary);
  ASSERT_TRUE(group->BindResource(0, depth.Get()));
  ASSERT_TRUE(group->BindResource(2, ordinary.Get()));
  ASSERT_TRUE(group->BindResource(3, output.Get()));
  ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  SamplerDesc desc{SAMPLER_FILTER_NEAREST};
  desc.compare_enable = true;
  desc.compare_op = static_cast<CompareOp>(99);
  EXPECT_FALSE(device_->CreateSampler(desc));
  const float expected[][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  for (int op = COMPARE_NEVER; op <= COMPARE_ALWAYS; ++op) {
    desc.compare_op = static_cast<CompareOp>(op);
    ObjPtr<ISampler> sampler = device_->CreateSampler(desc);
    ASSERT_TRUE(sampler);
    ASSERT_TRUE(group->BindResource(1, sampler.Get()));
    ASSERT_TRUE(group->BindResource(4, std::vector<ISampler *>{sampler.Get(), sampler.Get()}));
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands);
    RenderingInfo clear;
    clear.depth_stencil.image = depth.Get();
    clear.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
    clear.depth_stencil.clear_value.depth = .5f;
    ASSERT_TRUE(commands->CmdBeginRendering(clear));
    ASSERT_TRUE(commands->CmdEndRendering());
    ASSERT_TRUE(commands->CmdBindComputeProgram(program.Get()));
    ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
    ASSERT_TRUE(commands->CmdDispatch(3));
    ObjPtr<IDownloadToken> download = commands->CmdDownloadBuffer(output.Get(), 0, 3 * 4 * sizeof(float));
    ASSERT_TRUE(download);
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done && done->WaitHost());
    float values[3][4]{};
    ASSERT_TRUE(download->CopyTo(values, sizeof(values)));
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(values[i][0], expected[op][i]) << "op " << op << ", reference index " << i;
      EXPECT_FLOAT_EQ(values[i][1], .5f);
      EXPECT_FLOAT_EQ(values[i][2], expected[op][2]);
      EXPECT_FLOAT_EQ(values[i][3], expected[op][0]);
    }
  }
}

TEST_P(RhiPCF, RejectsSamplingTheActiveDepthAttachment) {
  ObjPtr<IImage> depth = device_->CreateImage({16, 16, IMAGE_FORMAT_D32_SFLOAT});
  ASSERT_TRUE(depth);
  const std::string source = R"(
    RHI_TEXTURE2D(0,0,float,depth);
    RHI_COMPARISON_SAMPLER(0,1,comparison);
    float4 VSMain(uint id:SV_VertexID):SV_Position {
      float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
      return float4(p[id],.5,1);
    }
    float PSMain():SV_Depth { return depth.SampleCmpLevelZero(comparison,float2(.5,.5),.5); }
  )";
  ObjPtr<IShader> vs = device_->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> ps = device_->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  ASSERT_TRUE(vs && ps);
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.color_formats.clear();
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = desc.depth_stencil.depth_write = true;
  desc.pipeline_layout.groups = {
      {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device_->CreateProgram(desc);
  ObjPtr<IBindingGroup> group = device_->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  SamplerDesc sampling{SAMPLER_FILTER_NEAREST};
  sampling.compare_enable = true;
  ObjPtr<ISampler> sampler = device_->CreateSampler(sampling);
  ASSERT_TRUE(program && group && sampler);
  ASSERT_TRUE(group->BindResource(0, depth.Get()));
  ASSERT_TRUE(group->BindResource(1, sampler.Get()));
  ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
  ASSERT_TRUE(commands);
  ASSERT_TRUE(commands->CmdBindProgram(program.Get()));
  ASSERT_TRUE(commands->CmdBindGroup(0, group.Get()));
  RenderingInfo render;
  render.depth_stencil.image = depth.Get();
  render.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  ASSERT_TRUE(commands->CmdBeginRendering(render));
  ASSERT_TRUE(commands->CmdSetViewport({0, 0, 16, 16, 0, 1}));
  ASSERT_TRUE(commands->CmdSetScissor({0, 0, 16, 16}));
  EXPECT_FALSE(commands->CmdDraw(3));
  EXPECT_FALSE(commands->Submit());
}

TEST_P(RhiPCF, DemoRendersHardAndFilteredShadows) {
  rhi_pcf_demo::Scene scene;
  ASSERT_TRUE(scene.Initialize(device_.Get(), IMAGE_FORMAT_R8G8B8A8_UNORM));
  ObjPtr<IQueue> queue = device_->GetQueue(QUEUE_TYPE_MAIN);
  ASSERT_TRUE(queue);
  // Render the same scene normally, then at a second size with visibility shown.
  for (bool visibility : {false, true}) {
    const uint32_t width = visibility ? 960 : 1440, height = visibility ? 640 : 900;
    ObjPtr<IImage> target = device_->CreateImage({width, height, IMAGE_FORMAT_R8G8B8A8_UNORM});
    ASSERT_TRUE(target);
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    ASSERT_TRUE(commands && scene.Draw(commands.Get(), target.Get(), visibility));
    ObjPtr<IDownloadToken> download = commands->CmdDownloadImage(target.Get());
    ASSERT_TRUE(download);
    ObjPtr<ISubmissionToken> done = commands->Submit();
    ASSERT_TRUE(done && done->WaitHost());
    std::vector<uint8_t> pixels(width * height * 4);
    ASSERT_TRUE(download->CopyTo(pixels.data(), pixels.size()));
    if (visibility) {
      uint32_t dark[2]{}, light[2]{}, soft[2]{};
      for (uint32_t y = 72; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
          const auto i = (y * width + x) * 4, side = x / (width / 2);
          if (pixels[i] != pixels[i + 1] || pixels[i] != pixels[i + 2])
            continue;
          dark[side] += pixels[i] < 3;
          light[side] += pixels[i] > 252;
          soft[side] += pixels[i] > 8 && pixels[i] < 247;
        }
      for (int side = 0; side < 2; ++side) {
        EXPECT_GT(dark[side], 100u);
        EXPECT_GT(light[side], 1000u);
      }
      EXPECT_EQ(soft[0], 0u);
      EXPECT_GT(soft[1], 100u);
    }
    if (const char *directory = std::getenv("LONGMARCH_RHI_CAPTURE_DIR")) {
      std::ofstream file(
          std::string(directory) + "/pcf-" + BackendAPIName(GetParam()) + (visibility ? "-visibility.ppm" : ".ppm"),
          std::ios::binary);
      ASSERT_TRUE(file);
      file << "P6\n" << width << ' ' << height << "\n255\n";
      for (size_t i = 0; i < pixels.size(); i += 4)
        file.write(reinterpret_cast<const char *>(pixels.data() + i), 3);
      ASSERT_TRUE(file);
    }
  }
}

namespace {
const std::vector<BackendAPI> kPCFBackends{
#if defined(LONGMARCH_D3D12_ENABLED)
    BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    BACKEND_API_METAL,
#endif
};
}  // namespace

INSTANTIATE_TEST_SUITE_P(Backends, RhiPCF, testing::ValuesIn(kPCFBackends));
}  // namespace grassland::rhi
