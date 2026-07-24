cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(VLM_VIDEO_NPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fexceptions -pthread -O3")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

set(THIRD_PARTY_LIB "${CMAKE_SOURCE_DIR}/third_party/lib")
set(FFMPEG_BIN_PATH "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-rockchip/bin" CACHE PATH "Path to ffmpeg-rockchip binaries")

set(VLM_CORE_SOURCES
    src/core/image_preprocess.cpp
    src/core/frame_extractor.cpp
    src/core/vision_encoder.cpp
    src/core/llm_runtime.cpp
    src/core/text_util.cpp
    src/pipeline/audio_transcriber.cpp
    src/pipeline/video_context_pipeline.cpp
    src/runtime/service_state.cpp
    src/runtime/model_registry.cpp
    src/api/openai_handlers.cpp
)

add_library(vlm_core STATIC ${VLM_CORE_SOURCES})
target_include_directories(vlm_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/rkllm
    ${OPENCV_INCLUDE_DIRS}
)
target_compile_definitions(vlm_core PUBLIC FFMPEG_BIN_PATH="${FFMPEG_BIN_PATH}")
target_link_libraries(vlm_core PUBLIC
    ${OPENCV_LIBRARIES}
    ${THIRD_PARTY_LIB}/librknnrt.so
    ${THIRD_PARTY_LIB}/librkllmrt.so
    pthread
)

add_executable(vlm_cli src/app/cli_main.cpp)
target_link_libraries(vlm_cli PRIVATE vlm_core)
set_target_properties(vlm_cli PROPERTIES
    BUILD_RPATH "${THIRD_PARTY_LIB}"
    INSTALL_RPATH "${THIRD_PARTY_LIB}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "VLM_VIDEO_NPU"
)

add_executable(vlm_api_server
    src/app/api_main.cpp
    src/api/http_server.cpp
)
target_link_libraries(vlm_api_server PRIVATE vlm_core)
set_target_properties(vlm_api_server PROPERTIES
    BUILD_RPATH "${THIRD_PARTY_LIB}"
    INSTALL_RPATH "${THIRD_PARTY_LIB}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

enable_testing()

add_executable(test_model_registry tests/test_model_registry.cpp)
target_link_libraries(test_model_registry PRIVATE vlm_core)
add_test(NAME test_model_registry COMMAND test_model_registry)

add_executable(test_text_util tests/test_text_util.cpp)
target_link_libraries(test_text_util PRIVATE vlm_core)
add_test(NAME test_text_util COMMAND test_text_util)

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE vlm_core)
add_test(NAME test_config COMMAND test_config)

add_executable(test_thinking_control tests/test_thinking_control.cpp)
target_link_libraries(test_thinking_control PRIVATE vlm_core)
add_test(NAME test_thinking_control COMMAND test_thinking_control)

add_executable(test_frame_plan tests/test_frame_plan.cpp)
target_link_libraries(test_frame_plan PRIVATE vlm_core)
add_test(NAME test_frame_plan COMMAND test_frame_plan)

add_executable(test_workdir tests/test_workdir.cpp)
target_link_libraries(test_workdir PRIVATE vlm_core)
add_test(NAME test_workdir COMMAND test_workdir)
