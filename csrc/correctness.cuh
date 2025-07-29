/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 5/17/25.
//

#ifndef CORRECTNESS_CUH
#define CORRECTNESS_CUH
#include <matx.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "include/kleos/telemetry.cuh"
#include "include/kleos/types.cuh"
#include "include/kleos/debug.cuh"

namespace kleos {
    // reference expert
    template<
        unsigned int S,
        unsigned int H,
        unsigned int P,
        unsigned int E,
        typename Element
    >
    __host__ __forceinline__
    void rExpert(Element* __restrict__ const& act,
        Element* __restrict__ const& gateWeights,
        Element* __restrict__ const& expertWeights,
        Element* __restrict__ const& bias,
        Element* __restrict__ const& gateOutput,
        Element* __restrict__ const& moeOutput,
        const unsigned int& nLx) {
        auto a = matx::make_tensor<Element>(act, {S, H});
        auto gW = matx::make_tensor<Element>(gateWeights, {H, E});
        auto gO = matx::make_tensor<Element>(gateOutput, {S, E});
        auto t0 = matx::make_tensor<Element>({});
        auto t0i = matx::make_tensor<matx::index_t>({});
        matx::cudaExecutor exec{kleosStream};
        // do Gate
        // 1) GEMM + Softmax
        (gO = matx::softmax(matx::matmul(a, gW), {1})).run(exec);
        (matx::mtie(t0, t0i) = matx::argmax(gO, {1})).run(exec);
    }

    __host__ __forceinline__
    void runReference() {
        constexpr auto S = 32;
        constexpr auto H = 32;
        constexpr auto E = 16;
        constexpr auto P = 32;
        constexpr auto PX = E;
        constexpr unsigned long aZ =  S * H;
        constexpr auto gwZ = aZ + PX * H;
        // scale this to number of experts
        constexpr auto nLx = E;
        constexpr auto bZ =  gwZ + nLx * P * H;
        constexpr auto b2Z =  bZ + nLx * P * H;
        constexpr auto dZ =  b2Z + nLx * (P + H);
        constexpr auto gZ = dZ + S * PX;
        constexpr auto cZ = gZ + S * H;
        void* p;
        KLEOS_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), kleos::kleosStream));
        KLEOS_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), kleos::kleosStream));
        auto* hP = std::calloc(cZ, sizeof(float));
        auto* fHp = static_cast<float*>(hP);
        using ET = float;
        auto* __restrict__ eHp = static_cast<ET*>(hP);
        auto* __restrict__ dP = static_cast<ET*>(p);
        {
            #if KLEOS_NVTX
            kleos::kleosRange forwardRange{"Host Data Prep"};
            #endif
            thrust::default_random_engine rng(47 * 42);
            thrust::normal_distribution<float> dist(0, 5);
            // Activations, Gate weights, expert weights
            thrust::generate(fHp, fHp + b2Z, [&] { return dist(rng); });
            if constexpr (!cuda::std::is_same_v<ET, float>) {
                constexpr cutlass::NumericConverter<ET, float> conv{};
                for (uint i = 0; i < dZ; ++i) {
                    eHp[i] = conv(fHp[i]);
                }
            }
        }
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(ET) * dZ,
            cudaMemcpyHostToDevice,
            kleos::kleosStream));
        auto* __restrict__ act = dP;
        auto* __restrict__ gateWeights = dP + aZ;
        auto* __restrict__ expertWeights = dP + gwZ;
        auto* __restrict__ bias = dP + b2Z;
        auto* __restrict__ gateOutput = dP + dZ;
        auto* __restrict__ moeOutput = dP + gZ;
        kleos::rExpert<S, H, P, E>(act,
            gateWeights, expertWeights, bias, gateOutput, moeOutput, nLx);
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(eHp, gateOutput, sizeof(ET) * S * PX, cudaMemcpyDeviceToHost,
            kleos::kleosStream));
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleos::kleosStream));
        const auto cGo = make_tensor(eHp,
                cute::Layout<cute::Shape<cute::Int<S>, cute::Int<E>>,
                cute::Stride<cute::Int<E>, cute::_1>>{});
        print_tensor(cGo);
    }
}
#endif //CORRECTNESS_CUH
