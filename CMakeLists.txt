cmake_minimum_required(VERSION 3.14)

project(hpolib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU ENVIRONMENT
option(HPOLIB_ENABLE_CUDA "Enable CUDA support for GPU acceleration" ON)
if(HPOLIB_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found, enabling GPU support.")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # Only auto-detect architectures if the user did NOT explicitly set them.
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            find_program(NVIDIA_SMI nvidia-smi)
            if (NVIDIA_SMI)
                execute_process(
                    COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE GPU_COMPUTE_CAP_RAW
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )

                if (GPU_COMPUTE_CAP_RAW)
                    # split lines into list (one entry per GPU)
                    string(REPLACE "\r\n" "\n" GPU_COMPUTE_CAP_RAW "${GPU_COMPUTE_CAP_RAW}")
                    string(REPLACE "\n" ";" GPU_COMPUTE_CAP_LIST "${GPU_COMPUTE_CAP_RAW}")

                    set(_normalized_archs "")
                    foreach(_cap IN LISTS GPU_COMPUTE_CAP_LIST)
                        # keep only digits and dot, then remove dot: "8.0" -> "80", "11.8" -> "118"
                        string(REGEX REPLACE "[^0-9\\.]" "" _cap_clean "${_cap}")
                        if (_cap_clean)
                            string(REPLACE "." "" _cap_norm "${_cap_clean}")
                            if (_cap_norm)
                                list(APPEND _normalized_archs "${_cap_norm}")
                            endif()
                        endif()
                    endforeach()

                    # remove duplicates
                    list(REMOVE_DUPLICATES _normalized_archs)

                    if (_normalized_archs)
                        message(STATUS "Detected GPU compute capability(s): ${GPU_COMPUTE_CAP_RAW}")
                        message(STATUS "Normalized CUDA architectures: ${_normalized_archs}")
                        # Save into cache so user can see/override later
                        set(CMAKE_CUDA_ARCHITECTURES "${_normalized_archs}" CACHE STRING "CUDA architectures to build for (normalized from nvidia-smi)")
                    else()
                        message(WARNING "Could not parse compute_cap from nvidia-smi: '${GPU_COMPUTE_CAP_RAW}'. You can set -DCMAKE_CUDA_ARCHITECTURES=80 (or similar) manually.")
                    endif()
                else()
                    message(WARNING "nvidia-smi returned empty compute capability. You can set -DCMAKE_CUDA_ARCHITECTURES=80 manually.")
                endif()
            else()
                message(WARNING "nvidia-smi not found in PATH; cannot auto-detect GPU compute capability. Set -DCMAKE_CUDA_ARCHITECTURES=80 (or desired arches) manually.")
            endif()
        else()
            message(STATUS "Using user-provided CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        message(STATUS "CUDA Toolkit not found, disabling GPU support. Building in CPU-only mode.")
        set(HPOLIB_ENABLE_CUDA OFF)
    endif()
endif()

add_library(hpolib INTERFACE)

target_include_directories(hpolib INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>
    $<INSTALL_INTERFACE:csrc>
)

enable_testing()

message(STATUS "Configuring GoogleTest...")
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
FetchContent_MakeAvailable(googletest)
message(STATUS "GoogleTest configured.")

add_subdirectory(tests)