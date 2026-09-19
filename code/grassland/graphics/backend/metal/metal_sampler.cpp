#include "grassland/graphics/backend/metal/metal_sampler.h"

#include "grassland/graphics/backend/metal/metal_core.h"

namespace grassland::graphics::backend {

MetalSampler::MetalSampler(MetalCore *core, const SamplerInfo &info) {
  MetalPool pool;
  auto descriptor = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
  descriptor->setMinFilter(info.min_filter == FILTER_MODE_LINEAR ? MTL::SamplerMinMagFilterLinear
                                                                 : MTL::SamplerMinMagFilterNearest);
  descriptor->setMagFilter(info.mag_filter == FILTER_MODE_LINEAR ? MTL::SamplerMinMagFilterLinear
                                                                 : MTL::SamplerMinMagFilterNearest);
  descriptor->setMipFilter(info.mip_filter == FILTER_MODE_LINEAR ? MTL::SamplerMipFilterLinear
                                                                 : MTL::SamplerMipFilterNearest);
  auto address = [](AddressMode mode) {
    switch (mode) {
      case ADDRESS_MODE_REPEAT:
        return MTL::SamplerAddressModeRepeat;
      case ADDRESS_MODE_MIRRORED_REPEAT:
        return MTL::SamplerAddressModeMirrorRepeat;
      case ADDRESS_MODE_CLAMP_TO_BORDER:
        return MTL::SamplerAddressModeClampToBorderColor;
      default:
        return MTL::SamplerAddressModeClampToEdge;
    }
  };

  descriptor->setSAddressMode(address(info.address_mode_u));
  descriptor->setTAddressMode(address(info.address_mode_v));
  descriptor->setRAddressMode(address(info.address_mode_w));
  descriptor->setSupportArgumentBuffers(true);
  state = NS::TransferPtr(core->Device()->newSamplerState(descriptor.get()));
  MetalCheck(state.get(), nullptr, "newSamplerState");
}

}  // namespace grassland::graphics::backend
