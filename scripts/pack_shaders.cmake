
set(LONG_MARCH_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Grassland Source Dir")
set(LONG_MARCH_INCLUDE_DIR ${LONG_MARCH_SOURCE_DIR}/src CACHE STRING "Grassland Include Dir")
set(LONG_MARCH_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "Grassland Binary Dir")
set(LONG_MARCH_PYTHON3_EXECUTABLE ${Python3_EXECUTABLE} CACHE STRING "Grassland Python3 Executable")

if (WIN32)
    # Define a cmake function XXD
    function(XXD input_file output_file dir_name)
        add_custom_command(
                OUTPUT ${output_file}
                COMMAND powershell -ExecutionPolicy Bypass -File ${LONG_MARCH_SOURCE_DIR}/scripts/xxd.ps1 ${input_file} ${output_file}
                COMMAND ${CMAKE_COMMAND} -E echo "Generating ${output_file} from ${input_file}"
                WORKING_DIRECTORY ${dir_name}
                DEPENDS ${input_file}
        )
        message(STATUS "XXD ${input_file} ${output_file} ${dir_name}")
    endfunction()
else ()
    # Define a cmake function XXD for Unix-Like systems
    # Run in the directory with file name as input file only
    function(XXD input_file output_file dir_name)
        # Add command with relative path
        add_custom_command(
                OUTPUT ${output_file}
                COMMAND xxd -i ${input_file} ${output_file}
                COMMAND ${CMAKE_COMMAND} -E echo "Generating ${output_file} from ${input_file}"
                WORKING_DIRECTORY ${dir_name}
                DEPENDS ${input_file}
        )
    endfunction()
endif ()

function(flatten_glsl_shader input_file output_file)
    # Add a custom command to flatten the GLSL shader
    # Get directory of the input file
    # Find all .glsl under the input file directory
    file(GLOB_RECURSE SHADER_INCLUDE_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/*.glsl
    )

    # Get directory of the output file
    get_filename_component(OUTPUT_DIR ${output_file} DIRECTORY)

    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    set(SHADER_DEPEND_FILES ${input_file})
    list(APPEND SHADER_DEPEND_FILES ${SHADER_INCLUDE_FILES})
    message(STATUS "SHADER DEPEND FILES: ${SHADER_DEPEND_FILES}")

    add_custom_command(
            OUTPUT ${output_file}
            # Make dir of OUTPUT_DIR
            COMMAND ${LONG_MARCH_PYTHON3_EXECUTABLE} ${LONG_MARCH_SOURCE_DIR}/scripts/flatten_glsl.py ${input_file} ${output_file}
            DEPENDS ${SHADER_DEPEND_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Flattening GLSL shader: ${input_file} -> ${output_file}"
    )
endfunction()

function(PACK_SHADER_CODE TARGET_NAME)
    message(STATUS "PACK_SHADER_CODE ${CMAKE_CURRENT_SOURCE_DIR}")

    # Find all the shader files under current directory
    file(GLOB_RECURSE SHADER_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/*.vert
            ${CMAKE_CURRENT_SOURCE_DIR}/*.frag
            ${CMAKE_CURRENT_SOURCE_DIR}/*.tesc
            ${CMAKE_CURRENT_SOURCE_DIR}/*.tese
            ${CMAKE_CURRENT_SOURCE_DIR}/*.comp
            ${CMAKE_CURRENT_SOURCE_DIR}/*.geom
            ${CMAKE_CURRENT_SOURCE_DIR}/*.rgen
            ${CMAKE_CURRENT_SOURCE_DIR}/*.rchit
            ${CMAKE_CURRENT_SOURCE_DIR}/*.rmiss
            ${CMAKE_CURRENT_SOURCE_DIR}/*.hlsl
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            LIST_DIRECTORIES false
    )

    # Extract shader files the relative path from the current source directory
    foreach (SHADER_FILE ${SHADER_FILES})
        file(RELATIVE_PATH RELATIVE_SHADER_FILE ${CMAKE_CURRENT_SOURCE_DIR} ${SHADER_FILE})
        list(APPEND RELATIVE_SHADER_FILES ${RELATIVE_SHADER_FILE})
    endforeach ()

    set(SHADER_FILES ${RELATIVE_SHADER_FILES})

    # Show all SHADER_FILES
    foreach (SHADER_FILE ${SHADER_FILES})
        message(STATUS "SHADER_FILE ${SHADER_FILE}")
    endforeach ()

    # Flatten all the shader files
    foreach (SHADER_FILE ${SHADER_FILES})
        set(FLATTENED_SHADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_FILE})
        flatten_glsl_shader(${SHADER_FILE} ${FLATTENED_SHADER_FILE})
        list(APPEND FLATTENED_SHADER_FILES ${FLATTENED_SHADER_FILE})
    endforeach ()

    # Make the flattened shaders a target
    add_custom_target(
            ${TARGET_NAME}_flattened_shaders ALL
            DEPENDS ${FLATTENED_SHADER_FILES}
    )

    # Use the XXD cmake function generate header files in corresponding build directory
    foreach (SHADER_FILE ${SHADER_FILES})
        set(HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_FILE}.h)
        # Use the corresponding flattened shader file as input
        set(FLATTENED_SHADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_FILE})
        XXD(${SHADER_FILE} ${HEADER_FILE} ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND HEADER_FILES ${HEADER_FILE})
    endforeach ()

    # Add the generated header files to the target
    target_sources(${TARGET_NAME} PRIVATE ${HEADER_FILES})

    # The custom command should be executed before the target is built
    add_custom_target(
            ${TARGET_NAME}_shader_files ALL
            DEPENDS ${HEADER_FILES}
    )

    add_dependencies(${TARGET_NAME}_shader_files ${TARGET_NAME}_flattened_shaders)

    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_shader_files)

    # Output the generated header files to built_in_shaders.inl in form #include<PATH_HEADER>

    # Get the relative path of the generated header files
    foreach (HEADER_FILE ${HEADER_FILES})
        file(RELATIVE_PATH RELATIVE_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR} ${HEADER_FILE})
        list(APPEND RELATIVE_HEADER_FILES ${RELATIVE_HEADER_FILE})
    endforeach ()

    # Generate the built_in_shaders.inl
    set(BUILT_IN_SHADERS_INL ${CMAKE_CURRENT_BINARY_DIR}/built_in_shaders.inl)
    file(WRITE ${BUILT_IN_SHADERS_INL} "// This file is generated by CMake\n")
    foreach (RELATIVE_HEADER_FILE ${RELATIVE_HEADER_FILES})
        file(APPEND ${BUILT_IN_SHADERS_INL} "#include \"${RELATIVE_HEADER_FILE}\"\n")
    endforeach ()

    # List all the shader info in a map std::map<std::string, std::pair<const char *, unsigned int>>, generate a global variable
    file(APPEND ${BUILT_IN_SHADERS_INL} "\n")
    file(APPEND ${BUILT_IN_SHADERS_INL} "std::map<std::string, std::pair<const char *, unsigned int>> shader_list = {\n")
    foreach (SHADER_FILE ${SHADER_FILES})
        string(REPLACE "." "_" VAR_NAME ${SHADER_FILE})
        string(REPLACE "/" "_" VAR_NAME ${VAR_NAME})
        string(REPLACE "\\" "_" VAR_NAME ${VAR_NAME})
        # Reintercast the variable name to const char *
        file(APPEND ${BUILT_IN_SHADERS_INL} "    {\"${SHADER_FILE}\", {reinterpret_cast<const char *>(${VAR_NAME}), ${VAR_NAME}_len}},\n")
    endforeach ()
    file(APPEND ${BUILT_IN_SHADERS_INL} "};\n")


    # Generate a function GetShaderCode in built_in_shaders.inl, input is the file name, output is the shader code, both are string
    file(APPEND ${BUILT_IN_SHADERS_INL} "\n")
    file(APPEND ${BUILT_IN_SHADERS_INL} "std::string GetShaderCode(const std::string& file_name) {\n")
    file(APPEND ${BUILT_IN_SHADERS_INL} "    return std::string(shader_list[file_name].first, shader_list[file_name].second);\n")
    file(APPEND ${BUILT_IN_SHADERS_INL} "}\n")


    target_sources(${TARGET_NAME} PRIVATE ${BUILT_IN_SHADERS_INL})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    message(STATUS "Including ${CMAKE_CURRENT_BINARY_DIR} in ${TARGET_NAME}")

endfunction()
