cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(VLM_VIDEO_NPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fexceptions -pthread -O3")

set(THIRD_PARTY_LIB "${CMAKE_SOURCE_DIR}/third_party/lib")
set(FFMPEG_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-rockchip/lib")
set(FFMPEG_INC_DIR "${CMAKE_SOURCE_DIR}/third_party/ffmpeg-rockchip/include")

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

set(VLM_GRPC_GEN_DIR "${CMAKE_SOURCE_DIR}/src/grpc_gen")
set(VLM_PROTO_FILE "${CMAKE_SOURCE_DIR}/proto/vlm/v1/worker.proto")

find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin
    HINTS /usr/bin /usr/local/bin)

if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "grpc_cpp_plugin not found; install protobuf-compiler-grpc")
endif()

set(VLM_GRPC_GEN_HDRS
    ${VLM_GRPC_GEN_DIR}/vlm/v1/worker.pb.h
    ${VLM_GRPC_GEN_DIR}/vlm/v1/worker.grpc.pb.h
)
set(VLM_GRPC_GEN_SRCS
    ${VLM_GRPC_GEN_DIR}/vlm/v1/worker.pb.cc
    ${VLM_GRPC_GEN_DIR}/vlm/v1/worker.grpc.pb.cc
)

add_custom_command(
    OUTPUT ${VLM_GRPC_GEN_HDRS} ${VLM_GRPC_GEN_SRCS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VLM_GRPC_GEN_DIR}
    COMMAND ${PROTOC_EXECUTABLE}
        -I ${CMAKE_SOURCE_DIR}/proto
        --cpp_out=${VLM_GRPC_GEN_DIR}
        --grpc_out=${VLM_GRPC_GEN_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
        ${VLM_PROTO_FILE}
    DEPENDS ${VLM_PROTO_FILE}
    COMMENT "Generating gRPC stubs from ${VLM_PROTO_FILE}"
)

set(VLM_GRPC_SOURCES ${VLM_GRPC_GEN_SRCS})

set(VLM_CORE_SOURCES
    src/core/audio_extractor.cpp
    src/core/frame_extractor.cpp
    src/core/vision_encoder.cpp
    src/core/vision_prompts.cpp
    src/core/llm_runtime.cpp
    src/core/subprocess.cpp
    src/core/text_util.cpp
    src/core/system_memory.cpp
    src/pipeline/audio_transcriber.cpp
    src/pipeline/video_context_pipeline.cpp
    src/pipeline/stage_transport.cpp
    src/pipeline/local_stage_transport.cpp
    src/pipeline/grpc_stage_transport.cpp
    src/grpc/tensor_codec.cpp
    src/grpc/grpc_client.cpp
    src/workers/grpc_workers.cpp
    ${VLM_GRPC_SOURCES}
    src/runtime/service_state.cpp
    src/runtime/job_progress.cpp
    src/runtime/model_registry.cpp
    src/api/openai_handlers.cpp
)

add_library(vlm_core STATIC ${VLM_CORE_SOURCES})
target_include_directories(vlm_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/rkllm
    ${CMAKE_SOURCE_DIR}/src/grpc_gen
    ${FFMPEG_INC_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)
target_link_directories(vlm_core PUBLIC ${FFMPEG_LIB_DIR})
target_compile_definitions(vlm_core PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
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
    ${Protobuf_LIBRARIES}
    ${GRPC_LIBRARIES}
)

set(VLM_RUNTIME_RPATH "${THIRD_PARTY_LIB};${FFMPEG_LIB_DIR}")

function(vlm_set_runtime_rpath target)
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

add_executable(vlm_vision_worker src/app/vision_worker_main.cpp)
target_link_libraries(vlm_vision_worker PRIVATE vlm_core)
vlm_set_runtime_rpath(vlm_vision_worker)
set_target_properties(vlm_vision_worker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(vlm_llm_worker src/app/llm_worker_main.cpp)
target_link_libraries(vlm_llm_worker PRIVATE vlm_core)
vlm_set_runtime_rpath(vlm_llm_worker)
set_target_properties(vlm_llm_worker PROPERTIES
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

add_executable(test_asr_segments tests/test_asr_segments.cpp)
target_link_libraries(test_asr_segments PRIVATE vlm_core)
vlm_set_runtime_rpath(test_asr_segments)
add_test(NAME test_asr_segments COMMAND test_asr_segments)

add_executable(test_system_memory tests/test_system_memory.cpp)
target_compile_definitions(test_system_memory PRIVATE
    TEST_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")
target_link_libraries(test_system_memory PRIVATE vlm_core)
vlm_set_runtime_rpath(test_system_memory)
add_test(NAME test_system_memory COMMAND test_system_memory)

add_executable(test_job_progress tests/test_job_progress.cpp)
target_link_libraries(test_job_progress PRIVATE vlm_core)
vlm_set_runtime_rpath(test_job_progress)
add_test(NAME test_job_progress COMMAND test_job_progress)

add_executable(test_api_auth tests/test_api_auth.cpp)
target_link_libraries(test_api_auth PRIVATE vlm_core)
vlm_set_runtime_rpath(test_api_auth)
add_test(NAME test_api_auth COMMAND test_api_auth)

add_executable(test_tensor_codec tests/test_tensor_codec.cpp)
target_link_libraries(test_tensor_codec PRIVATE vlm_core)
vlm_set_runtime_rpath(test_tensor_codec)
add_test(NAME test_tensor_codec COMMAND test_tensor_codec)

add_executable(test_grpc_targets tests/test_grpc_targets.cpp)
target_link_libraries(test_grpc_targets PRIVATE vlm_core)
vlm_set_runtime_rpath(test_grpc_targets)
add_test(NAME test_grpc_targets COMMAND test_grpc_targets)
