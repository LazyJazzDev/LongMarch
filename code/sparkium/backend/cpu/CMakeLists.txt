target_sources(sparkium_backend PRIVATE cpu_device.cpp shader_execution.cpp cpu_thread_pool.cpp)

target_sources(sparkium_backend PRIVATE resource_factory.cpp)

target_sources(sparkium_backend PRIVATE
        cpu_buffer.cpp
        cpu_command_context.cpp
        cpu_image.cpp
        cpu_memory.cpp
        cpu_program.cpp
        cpu_sampler.cpp
        cpu_shader.cpp
        slang_compiler.cpp)
