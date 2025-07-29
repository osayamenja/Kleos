# Copyright (c) 2025, Osayamen Jonathan Aimuyo
# All rights reserved.
#
# This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
# See the LICENSE file in the root directory for full terms.

cmake_minimum_required(VERSION 3.27)

# Add COMMAND_ECHO STDOUT to see the commands in standard out
# WAR to use updated gcc on NERSC machines
if(DEFINED ENV{NERSC_HOST})
    execute_process(COMMAND "which" "g++"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE CPP_COMP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "which" "gcc"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE C_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_COMPILER ${CPP_COMP_PATH})
    set(CMAKE_C_COMPILER ${C_COMP_PATH})
endif ()

project(csrc CUDA CXX)

# flags
set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -Wall -Wextra")
set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -fno-strict-aliasing")
if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
    set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -m64")
endif ()
set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -Wno-unknown-pragmas -Wnull-dereference -Wnarrowing")
set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -Wno-switch -Wduplicated-branches -Wformat=2")
set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
set(KLEOS_CUDA_CXX_FLAGS "${KLEOS_CUDA_CXX_FLAGS} -Wno-sign-compare -v")
# Silence nvtx deprecation warning from legacy version used in libtorch

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KLEOS_CUDA_CXX_FLAGS}")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${KLEOS_CUDA_CXX_FLAGS}\"")

# Query number of SMs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(SM)

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)

add_executable(csrc main.cu
        include/kleos/os/subscriber.cuh
        include/kleos/moe/moe.cuh
        include/kleos/kleos.cuh
        include/kleos/os/processor/processor.cuh
        include/kleos/indexing.cuh
        include/kleos/types.cuh
        include/kleos/atomics.cuh
        include/kleos/os/decider/decider.cuh
        include/kleos/os/decider/comps/edge.cuh
        include/kleos/os/decider/comps/expert.cuh
        include/kleos/os/decider/comps/group.cuh
        include/kleos/os/decider/comps/worker.cuh
        include/kleos/os/decider/comps/args.cuh
        include/kleos/os/decider/comps/functions.cuh
        include/kleos/debug.cuh
        include/kleos/os/decider/comps/niche.cuh
        include/kleos/topo.cuh
        include/kleos/os/processor/mmaConfig.cuh
        include/kleos/bootstrap.cuh
        include/kleos/os/scheduler.cuh
        include/kleos/moe/gate.cuh
        include/kleos/os/processor/gemm.cuh
        include/kleos/os/packet.cuh
        eval.cuh
        include/kleos/arch.cuh
        include/kleos/os/os.cuh
        include/kleos/moe/expert.cuh
        include/kleos/throughput.cuh
        include/kleos/moe/fffn.cuh
        include/kleos/telemetry.cuh
        correctness.cuh
)
set_target_properties(csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS   ON
)

string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU 0 Compute Capability: ${COMPUTE_CAPABILITY}")
set(ENV{CUTLASS_NVCC_ARCHS} "${COMPUTE_CAPABILITY}")

set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
set(CCCL_ENABLE_UNSTABLE ON)
include(cmake/CPM.cmake)
target_link_libraries(csrc PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml CUDA::cublas CUDA::nvtx3)
target_link_libraries(csrc PRIVATE atomic)
# add dependencies
## This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 2.8.4
)
if(CCCL_ADDED)
    target_link_libraries(csrc PRIVATE CCCL::CCCL)
endif()

# cuBLASDx
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)
target_link_libraries(csrc PRIVATE mathdx::cublasdx)

#CUTLASS business
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG v3.9.2 # pin here to support cuBLASDx
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${COMPUTE_CAPABILITY}"
)
if(CUTLASS_ADDED)
    # header-only
    target_include_directories(csrc SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
endif ()

set(FMT_SYSTEM_HEADERS, ON)
CPMAddPackage(
        NAME FMT
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.1.4
        OPTIONS
        "FMT_SYSTEM_HEADERS=ON"
)
if(FMT_ADDED)
    target_link_libraries(csrc PRIVATE fmt::fmt)
endif ()

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.1-c-cpp
        GIT_SHALLOW TRUE
)
if(NVTX3_ADDED)
    target_link_libraries(csrc PRIVATE nvtx3-cpp)
    string(FIND "$ENV{CMAKE_PREFIX_PATH}" "nvtx3" INDEX)
    if(INDEX EQUAL -1)
        # append nvtx3 to prefix path to remove libtorch error
        set(ENV{CMAKE_PREFIX_PATH} "$ENV{CMAKE_PREFIX_PATH}:${NVTX3_SOURCE_DIR}")
    endif()
endif ()

# MatX
set(MATX_NVTX_FLAGS ON)
set(MATX_ENABLE_FILEIO ON)
CPMAddPackage(
        NAME MATX
        GITHUB_REPOSITORY NVIDIA/MatX
        GIT_TAG v0.9.1
        OPTIONS
)
if(MATX_ADDED)
    target_link_libraries(csrc PRIVATE matx::matx)
endif ()

## NVSHMEM
find_package(NVSHMEM REQUIRED HINTS "$ENV{NVSHMEM_HOME}/lib/cmake/nvshmem")
if(DEFINED ENV{NERSC_HOST} AND "$ENV{NERSC_HOST}" STREQUAL "perlmutter")
    target_link_libraries(csrc PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)
else ()
    target_link_libraries(csrc PRIVATE nvshmem::nvshmem)
endif ()

target_compile_options(csrc PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
)

# Pre-compile constants
# Read JSON file content
file(READ "kleos_config.json" KLEOS_CONFIG)

string(JSON CAP_FACTOR_V GET ${KLEOS_CONFIG} "capacity_factor")
message(STATUS "capacity_factor: ${CAP_FACTOR_V}")

string(JSON DROP_TOKENS_V GET ${KLEOS_CONFIG} "drop_tokens")
message(STATUS "drop_tokens: ${DROP_TOKENS_V}")

string(JSON E_TOP_K_V GET ${KLEOS_CONFIG} "expert_top_k")
message(STATUS "expert_top_k: ${E_TOP_K_V}")

string(JSON GLOBAL_BATCH_V GET ${KLEOS_CONFIG} "global_batch")
message(STATUS "global_batch: ${GLOBAL_BATCH_V}")

string(JSON IS_TRAINING_V GET ${KLEOS_CONFIG} "is_training")
message(STATUS "is_training: ${IS_TRAINING_V}")

string(JSON HIDDEN_ACT_V GET ${KLEOS_CONFIG} "hidden_act")
message(STATUS "hidden_act: ${HIDDEN_ACT_V}")

string(JSON HIDDEN_SIZE_V GET ${KLEOS_CONFIG} "hidden_size")
message(STATUS "hidden_size: ${HIDDEN_SIZE_V}")

string(JSON I_SIZE_V GET ${KLEOS_CONFIG} "intermediate_size")
message(STATUS "intermediate_size: ${I_SIZE_V}")

string(JSON MINI_BATCH_V GET ${KLEOS_CONFIG} "mini_batch")
message(STATUS "mini_batch: ${MINI_BATCH_V}")

string(JSON MOE_FREQ_V GET ${KLEOS_CONFIG} "moe_frequency")
message(STATUS "moe_frequency: ${MOE_FREQ_V}")

string(JSON NUM_EXPERTS_V GET ${KLEOS_CONFIG} "num_experts")
message(STATUS "num_experts: ${NUM_EXPERTS_V}")

string(JSON NUM_LAYERS_V GET ${KLEOS_CONFIG} "num_layers")
message(STATUS "num_layers: ${NUM_LAYERS_V}")

string(JSON SEQ_LEN_V GET ${KLEOS_CONFIG} "sequence_len")
message(STATUS "sequence_len: ${SEQ_LEN_V}")

string(JSON DTYPE_V GET ${KLEOS_CONFIG} "torch_dtype")
message(STATUS "torch_dtype: ${DTYPE_V}")

string(JSON VOCAB_SIZE_V GET ${KLEOS_CONFIG} "vocab_size")
message(STATUS "vocab_size: ${VOCAB_SIZE_V}")

target_compile_definitions(csrc
        PRIVATE
        KLEOS_ARCH=${GPU_ARCH}
        CAP_FACTOR=${CAP_FACTOR_V}
        DROP_TOKENS=${DROP_TOKENS_V}
        E_TOP_K=${E_TOP_K_V}
        GLOBAL_BATCH=${GLOBAL_BATCH_V}
        IS_TRAINING=${IS_TRAINING_V}
        HIDDEN_ACT=${HIDDEN_ACT_V}
        HIDDEN_SIZE=${HIDDEN_SIZE_V}
        I_SIZE=${I_SIZE_V}
        MOE_FREQ=${MOE_FREQ_V}
        MINI_BATCH=${MINI_BATCH_V}
        NUM_EXPERTS=${NUM_EXPERTS_V}
        NUM_LAYERS=${NUM_LAYERS_V}
        SEQ_LEN=${SEQ_LEN_V}
        DTYPE=${DTYPE_V}
        VOCAB_SIZE=${VOCAB_SIZE_V}
        NUM_SMS=${CUDA_NUM_SMS}
        KLEOS_NVTX=1
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(csrc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O0;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O0; -g; -G>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(csrc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=lto_${COMPUTE_CAPABILITY}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr; -diag-suppress 186>
    )
endif ()
