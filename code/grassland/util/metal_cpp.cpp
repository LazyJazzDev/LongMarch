// Share one metal-cpp implementation between graphics and RHI.
#if defined(LONGMARCH_METAL_ENABLED)
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#endif
