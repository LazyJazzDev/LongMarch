#include "grassland/graphics/backend/d3d12/d3d12_commands.h"

#include "grassland/graphics/backend/d3d12/d3d12_command_context.h"
#include "grassland/graphics/backend/d3d12/d3d12_image.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

D3D12CmdClearImage::D3D12CmdClearImage(D3D12Image *image,
                                       const ClearValue &clear_value)
    : image_(image), clear_value_(clear_value) {
}

void D3D12CmdClearImage::CompileCommand(
    D3D12CommandContext *context,
    ID3D12GraphicsCommandList *command_list) {
  if (IsDepthFormat(image_->Format())) {
    context->RequireImageState(command_list, image_->Image()->Handle(),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE);
    D3D12_CLEAR_VALUE clear_value;
    clear_value.Format = ImageFormatToDXGIFormat(image_->Format());
    clear_value.DepthStencil.Depth = clear_value_.depth.depth;
    clear_value.DepthStencil.Stencil = 0;
    command_list->ClearDepthStencilView(
        context->DSVHandle(image_->Image()->Handle()), D3D12_CLEAR_FLAG_DEPTH,
        clear_value.DepthStencil.Depth, clear_value.DepthStencil.Stencil, 0,
        nullptr);
  } else {
    context->RequireImageState(command_list, image_->Image()->Handle(),
                               D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CLEAR_VALUE clear_value;
    clear_value.Format = ImageFormatToDXGIFormat(image_->Format());
    clear_value.Color[0] = clear_value_.color.r;
    clear_value.Color[1] = clear_value_.color.g;
    clear_value.Color[2] = clear_value_.color.b;
    clear_value.Color[3] = clear_value_.color.a;
    command_list->ClearRenderTargetView(
        context->RTVHandle(image_->Image()->Handle()), clear_value.Color, 0,
        nullptr);
  }
}

D3D12CmdPresent::D3D12CmdPresent(D3D12Window *window, D3D12Image *image)
    : image_(image), window_(window) {
}

void D3D12CmdPresent::CompileCommand(D3D12CommandContext *context,
                                     ID3D12GraphicsCommandList *command_list) {
  auto gpu_descriptor = context->WriteDescriptor(image_);

  context->RequireImageState(command_list, image_->Image()->Handle(),
                             D3D12_RESOURCE_STATE_GENERIC_READ);

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      window_->CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT,
      D3D12_RESOURCE_STATE_RENDER_TARGET);

  command_list->ResourceBarrier(1, &barrier);

  auto root_signature =
      context->Core()->BlitPipeline()->root_signature->Handle();
  auto pso = context->Core()->BlitPipeline()->GetPipelineState(
      window_->SwapChain()->BackBufferFormat());
  Extent2D extent;
  extent.width = window_->GetWidth();
  extent.height = window_->GetHeight();
  command_list->SetPipelineState(pso->Handle());
  command_list->SetGraphicsRootSignature(root_signature);
  command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  D3D12_RECT scissor_rect = {0, 0, LONG(extent.width), LONG(extent.height)};
  command_list->RSSetScissorRects(1, &scissor_rect);
  D3D12_VIEWPORT viewport = {
      0.0f, 0.0f, FLOAT(extent.width), FLOAT(extent.height), 0.0f, 1.0f};
  command_list->RSSetViewports(1, &viewport);
  const auto rtv = context->RTVHandle(window_->CurrentBackBuffer());
  command_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
  command_list->SetGraphicsRootDescriptorTable(0, gpu_descriptor);
  command_list->DrawInstanced(6, 1, 0, 0);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      window_->CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET,
      D3D12_RESOURCE_STATE_PRESENT);
  command_list->ResourceBarrier(1, &barrier);
}

}  // namespace grassland::graphics::backend
