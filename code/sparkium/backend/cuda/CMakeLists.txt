target_sources(sparkium_backend PRIVATE cuda_device.cpp shader_execution.cpp device_memory.cpp)
if (LONGMARCH_ENABLE_OPTIX)
    target_sources(sparkium_backend PRIVATE optix_acceleration_structure.cpp optix_device.cpp optix_launch.cpp)
endif ()

target_sources(sparkium_backend PRIVATE resource_factory.cpp)

target_sources(sparkium_backend PRIVATE
        cuda_buffer.cpp
        cuda_command_context.cpp
        cuda_image.cpp
        cuda_memory.cpp
        cuda_program.cpp
        cuda_sampler.cpp
        cuda_shader.cpp
        slang_compiler.cpp)
