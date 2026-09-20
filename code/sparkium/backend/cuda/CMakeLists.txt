target_sources(sparkium_backend PRIVATE cuda_device.cpp cuda_shader.cpp cuda_memory.cpp)
if (LONGMARCH_ENABLE_OPTIX)
    target_sources(sparkium_backend PRIVATE native_acceleration_structure.cpp optix_device.cpp optix_launch.cpp)
endif ()
