#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <stdexcept>

#include "grassland/graphics/graphics.h"

namespace {
using namespace grassland::graphics;

TEST(Submission, InvalidTicketDoesNotMeanCompleted) {
  Submission invalid;
  EXPECT_FALSE(invalid.IsValid());
  EXPECT_THROW(invalid.IsComplete(), std::logic_error);
  EXPECT_THROW(invalid.Wait(), std::logic_error);
}

TEST(Submission, PollWaitAndSlotReusePreserveOldTickets) {
  detail::SubmissionQueue queue;
  bool ready = false;
  int waits = 0, callbacks = 0;
  auto ticket = queue.Track([&] { return ready; },
                            [&] {
                              ++waits;
                              ready = true;
                            },
                            {[&] { ++callbacks; }});
  detail::InFlightSlot slot;
  slot.Retire(ticket);
  auto copy = ticket;
  EXPECT_FALSE(ticket.IsComplete());
  auto next = queue.Track([] { return true; }, [] {}, {});
  EXPECT_THROW(slot.Retire(next), std::logic_error);
  slot.Wait();
  slot.Retire(next);
  queue.Collect();
  EXPECT_TRUE(copy.IsComplete());
  copy.Wait();
  EXPECT_EQ(waits, 1);
  EXPECT_EQ(callbacks, 1);
  EXPECT_EQ(ticket.QueueId(), next.QueueId());
  EXPECT_LT(ticket.Value(), next.Value());
}

TEST(Submission, ErrorsRemainObservableAndOtherWorkIsRetired) {
  detail::SubmissionQueue queue;
  int success = 0;
  auto failed = queue.Track([]() -> bool { throw std::runtime_error("device lost"); },
                            [] { throw std::runtime_error("device lost"); }, {[&] { ++success; }});
  auto completed = queue.Track([] { return true; }, [] {}, {[&] { ++success; }});
  EXPECT_THROW(queue.Collect(), std::runtime_error);
  EXPECT_TRUE(completed.IsComplete());
  EXPECT_THROW(failed.Wait(), std::runtime_error);
  EXPECT_THROW(failed.IsComplete(), std::runtime_error);
  EXPECT_EQ(success, 1);
}

TEST(Submission, CallbackFailureDoesNotSkipRemainingCallbacks) {
  detail::SubmissionQueue queue;
  int released = 0;
  auto ticket =
      queue.Track([] { return true; }, [] {}, {[] { throw std::runtime_error("callback"); }, [&] { ++released; }});
  EXPECT_THROW(ticket.Wait(), std::runtime_error);
  EXPECT_EQ(released, 1);
  EXPECT_THROW(ticket.Wait(), std::runtime_error);
  EXPECT_EQ(released, 1);
}

TEST(Submission, FailedNativeSubmitDoesNotLeavePendingWork) {
  detail::SubmissionQueue queue;
  int callbacks = 0;
  EXPECT_THROW(queue.Track([] { return false; }, [] { FAIL() << "must not wait for failed submit"; },
                           {[&] { ++callbacks; }}, [] { throw std::runtime_error("submit failed"); }),
               std::runtime_error);
  queue.WaitAll();
  EXPECT_EQ(callbacks, 0);
}

TEST(Submission, DroppedTicketsStillRetireAndSurvivingTicketsOutliveQueue) {
  Submission survivor;
  int callbacks = 0;
  {
    detail::SubmissionQueue queue;
    queue.Track([] { return false; }, [] {}, {[&] { ++callbacks; }});
    survivor = queue.Track([] { return false; }, [] {}, {[&] { ++callbacks; }});
  }
  EXPECT_EQ(callbacks, 2);
  EXPECT_TRUE(survivor.IsComplete());
}

void CheckComputeSubmissions(BackendAPI api) {
  std::unique_ptr<Core> core;
  ASSERT_EQ(CreateCore(api, {2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  EXPECT_THROW(core->Submit(nullptr), std::invalid_argument);
  std::unique_ptr<Buffer> input, output;
  ASSERT_EQ(core->CreateBuffer(256, BUFFER_TYPE_DYNAMIC, &input), 0);
  ASSERT_EQ(core->CreateBuffer(12 * sizeof(uint32_t), BUFFER_TYPE_STATIC, &output), 0);
  std::unique_ptr<Shader> shader;
  ASSERT_EQ(core->CreateShader(R"(
cbuffer Input : register(b0, space0) { uint index; uint value; };
RWStructuredBuffer<uint> output : register(u0, space1);
[numthreads(1,1,1)] void Main() { output[index] = value; }
)",
                               "Main", "cs_6_0", &shader),
            0);
  std::unique_ptr<ComputeProgram> program;
  ASSERT_EQ(core->CreateComputeProgram(shader.get(), &program), 0);
  program->AddResourceBinding(RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER, 1);
  program->Finalize();
  std::vector<Submission> tickets;
  int callbacks = 0;
  for (uint32_t i = 0; i < 12; ++i) {
    std::array<uint32_t, 64> data{i, 1000 + i};
    input->UploadData(data.data(), sizeof(data));
    std::unique_ptr<CommandContext> commands;
    ASSERT_EQ(core->CreateCommandContext(&commands), 0);
    commands->CmdBindComputeProgram(program.get());
    commands->CmdBindResources(0, {input.get()}, BIND_POINT_COMPUTE);
    commands->CmdBindResources(1, {output.get()}, BIND_POINT_COMPUTE);
    commands->CmdDispatch(1, 1, 1);
    commands->OnCompleted([&] { ++callbacks; });
    auto ticket = core->Submit(commands.get());
    EXPECT_THROW(core->Submit(commands.get()), std::logic_error);
    EXPECT_THROW(commands->OnCompleted([] {}), std::logic_error);
    EXPECT_EQ(core->CurrentSubmissionSlot(), (i + 1) % 2);
    if (!tickets.empty()) {
      EXPECT_EQ(ticket.QueueId(), tickets.back().QueueId());
      EXPECT_GT(ticket.Value(), tickets.back().Value());
    }
    tickets.push_back(ticket);
    // Overwrite the CPU shadow immediately. Submitted copies must be snapshots.
    data.fill(0);
    input->UploadData(data.data(), sizeof(data));
  }
  tickets.back().Wait();
  core->CollectCompletedSubmissions();
  EXPECT_EQ(callbacks, 12);
  for (auto &ticket : tickets)
    EXPECT_TRUE(ticket.IsComplete());
  std::array<uint32_t, 12> result{};
  output->DownloadData(result.data(), sizeof(result));
  for (uint32_t i = 0; i < result.size(); ++i)
    EXPECT_EQ(result[i], 1000 + i);
  // This stage requires caller-owned resources to survive GPU completion.
  program.reset();
  shader.reset();
  input.reset();
  output.reset();
  // Core destruction also drains work whose completion was never queried.
  std::unique_ptr<CommandContext> final_commands;
  ASSERT_EQ(core->CreateCommandContext(&final_commands), 0);
  final_commands->OnCompleted([&] { ++callbacks; });
  auto final_ticket = core->Submit(final_commands.get());
  final_commands.reset();
  core.reset();
  EXPECT_EQ(callbacks, 13);
  EXPECT_TRUE(final_ticket.IsComplete());
  for (auto &ticket : tickets)
    EXPECT_TRUE(ticket.IsComplete());
}

void CheckPresentationSubmissions(BackendAPI api) {
  if (!std::getenv("LONGMARCH_TEST_SUBMISSION_WINDOWS"))
    GTEST_SKIP() << "Set LONGMARCH_TEST_SUBMISSION_WINDOWS for presentation tests";
  std::unique_ptr<Core> core;
  ASSERT_EQ(CreateCore(api, {2, true}, &core), 0);
  ASSERT_EQ(core->InitializeLogicalDeviceAutoSelect(false), 0);
  std::unique_ptr<Window> window;
  ASSERT_EQ(core->CreateWindowObject(320, 240, "Submission slots", &window), 0);
  std::unique_ptr<Image> image;
  ASSERT_EQ(core->CreateImage(320, 240, IMAGE_FORMAT_R8G8B8A8_UNORM, &image), 0);
  Submission last;
  for (int i = 0; i < 12; ++i) {
    std::unique_ptr<CommandContext> commands;
    ASSERT_EQ(core->CreateCommandContext(&commands), 0);
    commands->CmdClearImage(image.get(), {{0.1f, 0.2f, 0.3f, 1.0f}});
    commands->CmdPresent(window.get(), image.get());
    last = core->Submit(commands.get());
  }
  last.Wait();
  core->CollectCompletedSubmissions();
  window->CloseWindow();
}

#ifdef LONGMARCH_VULKAN_ENABLED
TEST(SubmissionGPU, VulkanPresentationSlots) {
  CheckPresentationSubmissions(BACKEND_API_VULKAN);
}

TEST(SubmissionGPU, VulkanComputeWithoutFrames) {
  CheckComputeSubmissions(BACKEND_API_VULKAN);
}
#endif
#ifdef LONGMARCH_D3D12_ENABLED
TEST(SubmissionGPU, D3D12PresentationSlots) {
  CheckPresentationSubmissions(BACKEND_API_D3D12);
}

TEST(SubmissionGPU, D3D12ComputeWithoutFrames) {
  CheckComputeSubmissions(BACKEND_API_D3D12);
}
#endif
#ifdef LONGMARCH_METAL_ENABLED
TEST(SubmissionGPU, MetalComputeWithoutFrames) {
  CheckComputeSubmissions(BACKEND_API_METAL);
}
#endif
}  // namespace
