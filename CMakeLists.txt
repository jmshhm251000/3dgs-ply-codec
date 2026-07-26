cmake_minimum_required(VERSION 3.16)
project(plycodec CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emit build/compile_commands.json so clangd (nvim LSP) knows the flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Auto-collect every .cpp under src/ — no need to edit this file when you add one.
# CONFIGURE_DEPENDS makes CMake re-glob on build if the set of files changed
# (so a new src/foo.cpp gets picked up without manually re-running cmake).
file(GLOB PLYCODEC_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(plycodec ${PLYCODEC_SOURCES})

# Headers live in include/ and are pulled in via #include — they don't need to be
# listed for compilation. This makes them resolve from any TU.
target_include_directories(plycodec PRIVATE include)

# SIMD: AVX2 + FMA, for the quantizer kernels later.
if(MSVC)
  target_compile_options(plycodec PRIVATE /O2 /arch:AVX2)
else()
  # -gdwarf-4: debug symbols in a version VTune can read (GCC13 defaults to DWARF-5,
  # which VTune fails to symbolize). Zero runtime cost. Add -fno-omit-frame-pointer
  # temporarily when you need complete VTune call stacks / flame graphs.
  target_compile_options(plycodec PRIVATE -O3 -mavx2 -mfma -gdwarf-4)
endif()

# CUDA renderer (M1 distortion meter). Optional: only built where a CUDA
# compiler is found (needs MSVC host on Windows, not MinGW).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  add_executable(renderer src/renderer.cu src/plyparser.cpp)
  set_target_properties(renderer PROPERTIES CUDA_ARCHITECTURES 86)
  target_include_directories(renderer PRIVATE include third_party)
else()
  message(STATUS "CUDA compiler not found - skipping 'renderer' target")
endif()
