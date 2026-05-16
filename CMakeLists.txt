cmake_minimum_required(VERSION 3.16)
project(tiny_tts LANGUAGES CXX)

option(TINY_TTS_BUILD_INFERENCE "Build ONNX inference engine" ON)
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "Path to the ONNX Runtime installation root")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tiny_tts STATIC)
target_include_directories(tiny_tts
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_sources(tiny_tts
  PRIVATE
    src/cli_options.cpp
    src/text_frontend.cpp
    src/wav_writer.cpp)

if(TINY_TTS_BUILD_INFERENCE)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime)

  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64)

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to an ONNX Runtime installation path.")
  endif()

  target_sources(tiny_tts PRIVATE src/onnx_engine.cpp src/tiny_tts.cpp)
  target_include_directories(tiny_tts PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(tiny_tts PRIVATE ${ONNXRUNTIME_LIBRARY})

  add_executable(tiny-tts-cli src/main.cpp)
  target_link_libraries(tiny-tts-cli PRIVATE tiny_tts)
endif()

enable_testing()
add_executable(text_frontend_smoke tests/text_frontend_smoke.cpp)
target_include_directories(text_frontend_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(text_frontend_smoke PRIVATE TINY_TTS_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
target_link_libraries(text_frontend_smoke PRIVATE tiny_tts)
add_test(NAME text_frontend_smoke COMMAND text_frontend_smoke)

add_executable(cli_options_test tests/cli_options_test.cpp)
target_include_directories(cli_options_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cli_options_test PRIVATE tiny_tts)
add_test(NAME cli_options_test COMMAND cli_options_test)
