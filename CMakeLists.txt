cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(VLM_VIDEO_NPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fexceptions -pthread -O3")

set(THIRD_PARTY_LIB "${CMAKE_SOURCE_DIR}/third_party/lib")
set(FFMPEG_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-rockchip/lib")
set(FFMPEG_INC_DIR "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-rockchip/include")

set(VLM_CORE_SOURCES
    src/core/audio_extractor.cpp
    src/core/frame_extractor.cpp
    src/core/vision_encoder.cpp
    src/core/llm_runtime.cpp
    src/core/subprocess.cpp
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
    ${FFMPEG_INC_DIR}
)
target_link_directories(vlm_core PUBLIC ${FFMPEG_LIB_DIR})
target_link_libraries(vlm_core PUBLIC
    avcodec
    avformat
    avfilter
    avutil
    swscale
    swresample
    drm
    z
    opus
    mp3lame
    vorbis
    vorbisenc
    ssl
    crypto
    ${THIRD_PARTY_LIB}/librknnrt.so
    ${THIRD_PARTY_LIB}/librkllmrt.so
    pthread
)

set(VLM_RUNTIME_RPATH "${THIRD_PARTY_LIB};${FFMPEG_LIB_DIR}")

function(vlm_set_runtime_rpath target)
    # DT_RPATH (not DT_RUNPATH) so vendored libav transitive deps (e.g. libswresample)
    # resolve when the loader opens libavcodec.so from our ffmpeg-rockchip directory.
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${VLM_RUNTIME_RPATH}"
        INSTALL_RPATH "${VLM_RUNTIME_RPATH}"
    )
    target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endfunction()

add_executable(vlm_cli src/app/cli_main.cpp)
target_link_libraries(vlm_cli PRIVATE vlm_core)
vlm_set_runtime_rpath(vlm_cli)
set_target_properties(vlm_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "VLM_VIDEO_NPU"
)

add_executable(vlm_api_server
    src/app/api_main.cpp
    src/api/http_server.cpp
)
target_link_libraries(vlm_api_server PRIVATE vlm_core)
vlm_set_runtime_rpath(vlm_api_server)
set_target_properties(vlm_api_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

enable_testing()

add_executable(test_model_registry tests/test_model_registry.cpp)
target_link_libraries(test_model_registry PRIVATE vlm_core)
vlm_set_runtime_rpath(test_model_registry)
add_test(NAME test_model_registry COMMAND test_model_registry)

add_executable(test_text_util tests/test_text_util.cpp)
target_link_libraries(test_text_util PRIVATE vlm_core)
vlm_set_runtime_rpath(test_text_util)
add_test(NAME test_text_util COMMAND test_text_util)

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE vlm_core)
vlm_set_runtime_rpath(test_config)
add_test(NAME test_config COMMAND test_config)

add_executable(test_thinking_control tests/test_thinking_control.cpp)
target_link_libraries(test_thinking_control PRIVATE vlm_core)
vlm_set_runtime_rpath(test_thinking_control)
add_test(NAME test_thinking_control COMMAND test_thinking_control)

add_executable(test_frame_plan tests/test_frame_plan.cpp)
target_link_libraries(test_frame_plan PRIVATE vlm_core)
vlm_set_runtime_rpath(test_frame_plan)
add_test(NAME test_frame_plan COMMAND test_frame_plan)

add_executable(test_workdir tests/test_workdir.cpp)
target_link_libraries(test_workdir PRIVATE vlm_core)
vlm_set_runtime_rpath(test_workdir)
add_test(NAME test_workdir COMMAND test_workdir)

add_executable(test_rgb_frame tests/test_rgb_frame.cpp)
target_link_libraries(test_rgb_frame PRIVATE vlm_core)
vlm_set_runtime_rpath(test_rgb_frame)
add_test(NAME test_rgb_frame COMMAND test_rgb_frame)

add_executable(test_frame_extractor tests/test_frame_extractor.cpp)
target_compile_definitions(test_frame_extractor PRIVATE
    TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
target_link_libraries(test_frame_extractor PRIVATE vlm_core)
vlm_set_runtime_rpath(test_frame_extractor)
add_test(NAME test_frame_extractor COMMAND test_frame_extractor)

add_executable(test_audio_extractor tests/test_audio_extractor.cpp)
target_compile_definitions(test_audio_extractor PRIVATE
    TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
target_link_libraries(test_audio_extractor PRIVATE vlm_core)
vlm_set_runtime_rpath(test_audio_extractor)
add_test(NAME test_audio_extractor COMMAND test_audio_extractor)

add_executable(test_subprocess tests/test_subprocess.cpp)
target_link_libraries(test_subprocess PRIVATE vlm_core)
vlm_set_runtime_rpath(test_subprocess)
add_test(NAME test_subprocess COMMAND test_subprocess)

add_executable(test_api_auth tests/test_api_auth.cpp)
target_link_libraries(test_api_auth PRIVATE vlm_core)
vlm_set_runtime_rpath(test_api_auth)
add_test(NAME test_api_auth COMMAND test_api_auth)
