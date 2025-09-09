list(APPEND CHANGZHENG_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})
list(APPEND CHANGZHENG_INCLUDE_DIR ${INC_DIR})
set(CHANGZHENG_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)

add_library(ChangZheng chang_zheng.cpp)

add_subdirectory(cao_di)
add_subdirectory(xue_shan)
add_subdirectory(xing_huo)

target_include_directories(ChangZheng PUBLIC ${CHANGZHENG_INCLUDE_DIR})
target_link_libraries(ChangZheng PUBLIC CaoDi XueShan XingHuo)

pybind11_add_module(chang_zheng chang_zheng_pybind.cpp)
target_include_directories(chang_zheng PUBLIC ${CHANGZHENG_INCLUDE_DIR})
target_link_libraries(chang_zheng PUBLIC ChangZheng)

# copy all related dll files to the destination too.
install(TARGETS chang_zheng
        RUNTIME_DEPENDENCY_SET chang_zheng_deps
        LIBRARY DESTINATION ${Python3_SITEARCH})

install(
        RUNTIME_DEPENDENCY_SET chang_zheng_deps
        DESTINATION ${Python3_SITEARCH}
        PRE_EXCLUDE_REGEXES
        [=[api-ms-]=]
        [=[ext-ms-]=]
        [[kernel32\.dll]]
        [[libc\.so\..*]] [[libgcc_s\.so\..*]] [[libm\.so\..*]] [[libstdc\+\+\.so\..*]]
        [[python*]]
        POST_EXCLUDE_REGEXES
        [=[.*system32\/.*\.dll]=]
        [=[^\/(lib|usr\/lib|usr\/local\/lib)]=]
        DIRECTORIES "$<TARGET_FILE_DIR:chang_zheng>"
)
