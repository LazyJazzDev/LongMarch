#include <cmath>
#include <iostream>

#include "HdrPresentation.h"

using namespace grassland::graphics;

namespace {
void Near(double actual, double expected, double tolerance, const char *message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
    throw std::runtime_error(message);
}

double DecodePQ(double signal) {
  const double p = std::pow(signal, 32.0 / 2523.0);
  return 10000.0 *
         std::pow(std::max(p - 3424.0 / 4096.0, 0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * p), 16384.0 / 2610.0);
}
}  // namespace

int main() {
  try {
    std::unique_ptr<Core> core;
    if (CreateCore(BACKEND_API_VULKAN, Core::Settings{1, true}, &core) ||
        core->InitializeLogicalDeviceAutoSelect(false))
      throw std::runtime_error("Cannot create Vulkan core");
    std::unique_ptr<Image> source;
    core->CreateImage(5, 1, IMAGE_FORMAT_R32G32B32A32_SFLOAT, &source);
    const float input[20] = {0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 0, 0, 1, 10, 10, 10, 1};
    source->UploadData(input);
    longmarch::harmony::HdrPresentation presentation(dynamic_cast<backend::VulkanCore *>(core.get()));
    float output[20];
    presentation.Convert(source.get(), true, false)->DownloadData(output);
    for (int c = 0; c < 3; ++c) {
      Near(DecodePQ(output[c]), 0, .01, "PQ black is not black");
      Near(DecodePQ(output[4 + c]), 203, .2, "HDR reference white must be 203 nits");
      Near(DecodePQ(output[8 + c]), 609, .2, "HDR highlights were clipped or incorrectly encoded");
      Near(DecodePQ(output[16 + c]), 1000, .2, "Signal exceeds declared 1000-nit container");
    }
    Near(DecodePQ(output[12]), 127.363012, .2, "Rec.709 red to BT.2020 red mismatch");
    Near(DecodePQ(output[13]), 14.026691, .2, "Rec.709 red to BT.2020 green mismatch");
    Near(DecodePQ(output[14]), 3.327373, .2, "Rec.709 red to BT.2020 blue mismatch");
    presentation.Convert(source.get(), false, false)->DownloadData(output);
    Near(output[0], 0, .0001, "SDR black mismatch");
    Near(output[4], 1, .0001, "SDR white mismatch");
    Near(output[8], 1, .0001, "SDR highlight must be bounded");
    presentation.Convert(source.get(), true, true)->DownloadData(output);
    Near(DecodePQ(output[8]), 1000, .2, "NBody gamma decoding is missing");
    core->WaitGPU();
    std::cout << "PASS HDR10 PQ luminance, BT.2020 primaries, SDR fallback and NBody decoding\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
