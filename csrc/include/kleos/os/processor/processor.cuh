/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
//
// Created by osayamen on 7/13/24.
//

#ifndef KLEOS_COMPUTE_CUH
#define KLEOS_COMPUTE_CUH

#include <cutlass/array.h>
#include <cute/tensor.hpp>
#include <nvshmem.h>

#include "gemm.cuh"
#include "../../types.cuh"

namespace kleos::processor{
    enum class ReleaseType {
        stable,
        experimental
    };
    template<
        CombineMode c = CombineMode::single,
        unsigned int gM = BLOCK_M,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        class ScaleWeights,
        typename Element,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value + ACC::PeakHardware::spare::value,
        unsigned int wS = WARP_SIZE
    >
    requires(TensorValueType<Element> &&
            elems % wS == 0 && // guarantees warp convergence
            kleos::isMatrix<ScaleWeights> &&
            cuda::std::is_same_v<typename ScaleWeights::value_type, Element>)
    __device__ __forceinline__
    void combine(cuda::std::byte* __restrict__ const& workspace,
            const TPS* __restrict__ const& tokenIndices,
            const Element* __restrict__ const& inputs,
            Element* __restrict__ const& moeOutput,
            ScaleWeights const& scale,
            const unsigned int& tileIdx,
            const uint16_t& tileSize,
            const uint16_t& expertIdx) {
        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        constexpr BlockTiler tiler{};
        constexpr auto bM = cute::get<0>(tiler);
        constexpr auto bN = cute::get<1>(tiler);
        cutlass::Array<Element, bN> registers{};
        constexpr auto mTe = cutlass::NumericConverter<Element, mp_t>{};
        constexpr auto eTm = cutlass::NumericConverter<mp_t, Element>{};
        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            cute::Layout<cute::Shape<cute::Int<gM>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mC = make_tensor(cute::make_gmem_ptr(moeOutput),
            cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});

        // We assert the below prior to this point
        static_assert(gM % bM == 0);
        constexpr auto tilesM = gM / bM;
        constexpr auto tilesN = N / bN;

        const auto tileCoord = idx2crd(tileIdx,
            cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
        const auto gA = cute::local_tile(mA, tiler, ctaCoord);

        const auto tileCoordOut = idx2crd(tileIdx,
            cute::Shape<cute::_1, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto gC = cute::local_tile(mC,
            cute::Shape<cute::Int<M>, cute::Int<bN>>{},
                cute::make_coord(cute::get<0>(tileCoordOut),
                    cute::get<1>(tileCoordOut)));
        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;
        // ensures we have enough shared memory
        static_assert(sizeof(TPS) * bM <= sharedSize);
        static_assert(bM % elems == 0);
        // slice and dice token indices
        constexpr auto phases = bM / elems;
        const auto phaseIdx = threadIdx.x / elems;
        static_assert(elems % wS == 0);
        constexpr auto wE = elems / wS;
        auto* __restrict__ sTPS = CAST_TO(TPS, workspace);
        static_assert(bM == threads);
        sTPS[threadIdx.x] = tokenIndices[threadIdx.x];
        __syncthreads();
        // Eagerly prefetch inputs to registers
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            // global -> registers
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto rIdx = phaseIdx + j * phases;
                const auto cIdx =  threadIdx.x % elems + i * elems;
                registers[j + i * elems] = gA(rIdx, cIdx);
            }
        }
        if constexpr (c == CombineMode::multithreaded) {
            TPS wT[wE];
            Element rS[wE];
            #pragma unroll
            for (uint i = 0; i < wE; ++i) {
                const auto tid = threadIdx.x % wS;
                wT[i] = sTPS[phaseIdx + (tid + i * wS) * phases];
                rS[i] = scale(wT[i].tokenIdx, expertIdx);
            }
            __syncwarp();
            cutlass::Array<uint, elems> tIds{};
            using CDxT = typename ToCDx<Element>::T;
            constexpr auto cTCx = cutlass::NumericConverter<CDxT, Element>{};
            const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(CDxT, registers.data())),
                cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>, cute::Stride<cute::Int<bN>, cute::_1>>{});
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(ull_t, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                const auto [tokenIdx, probability] = *CONST_CAST_TO(TPS, &rM);
                tIds[j] = tokenIdx;
                // apply division operation
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    registers[i + j * elems] = mTe(__fdividef(eTm(registers[i + j * elems]), probability));
                }
            }
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = rS[j / wS];
                const auto* __restrict__ mP = CONST_CAST_TO(CDxT, &msg);
                const auto rM = __shfl_sync(0xffffffff, *mP, j % wS);
                // apply scale
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    rC(j + i * elems) = cTCx(*CONST_CAST_TO(Element, &rM) * registers[j + i * elems]);
                }
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // do atomic addition
                if (tileSize < gM) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        if (phaseIdx + j * phases < tileSize) {
                            atomicAdd(CAST_TO(CDxT, &gC(tIds[j], cIdx)), rC(j + i * elems));
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        const auto cIdx = threadIdx.x % elems + i * elems;
                        atomicAdd(CAST_TO(CDxT, &gC(tIds[j], cIdx)), rC(j + i * elems));
                    }
                }
            }
        }
        else {
            // scatter
            uint wT[wE];
            const auto tid = threadIdx.x % wS;
            #pragma unroll
            for (uint i = 0; i < wE; ++i) {
                wT[i] = sTPS[phaseIdx + (tid + i * wS) * phases].tokenIdx;
            }
            // vector copy from registers to global directly and call it a day
            cutlass::AlignedArray<uint, elems> tIds{};
            __syncwarp();
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                const auto msg = wT[j / wS];
                tIds[j] = __shfl_sync(0xffffffff, msg, j % wS);
            }
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                const auto cIdx = threadIdx.x % elems + i * elems;
                if (tileSize < gM) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        // predicated writes
                        if (phaseIdx + j * phases < tileSize) {
                            gC(tIds[j], cIdx) = registers[j + i * elems];
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        gC(tIds[j], cIdx) = registers[j + i * elems];
                    }
                }
            }
        }
    }

    // fused GEMM, epilogue and data transfer, with static M, N and K
    template<
        typename BlockGEMM,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        unsigned int K = ACC::P::value,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void sfGET(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<K>>,
            cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
        cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,
            cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
            cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
            cute::Layout<cute::Shape<cute::_1, cute::Int<N>>,
                cute::Stride<cute::_0, cute::_1>>{});

        // M is padded, such that the below is correct
        constexpr auto tilesM = M / bM;
        // We assert the below prior to this point
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});

        const auto k_tile_iter = cute::make_coord_iterator(tilesK);
        using Element = typename BlockGEMM::MatrixDType;

        // prefetch bias from global memory
        static_assert(ACC::sharedSize::value >= ACC::GSM::value + sizeof(Element) * bN);
        const auto biasCoord = idx2crd(tileIdx, cute::Shape<cute::_1, cute::Int<tilesN>>{},
            cute::Stride<cute::Int<bN>, cute::_1>{});
        const auto gD = cute::local_tile(mD,
            cute::Shape<cute::_1, cute::Int<bN>>{}, cute::get<1>(biasCoord));
        static_assert(threads % bN == 0);
        if (threadIdx.x < bN) {
            using LT =  cuda::std::conditional_t<sizeof(Element) == 2, uint16_t, uint32_t>;
            CAST_TO(LT, workspace)[threadIdx.x] = __ldg(CONST_CAST_TO(LT, &gD(threadIdx.x)));
        }
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace));
        /// There is a block-wide barrier at the end of the above ^

        // Epilogue
        using ElementC = typename decltype(accumulator)::value_type;
        typename BlockGEMM::MMA tiledMMA{};
        constexpr auto gCStoreOp = cutlass::NumericConverter<Element, ElementC>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, Element>{};
        // Assume elementwise operator
        typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / elems;
        // copy single bias value
        ElementC rB[trips];
        #pragma unroll
        for (uint i = 0 ; i < trips; ++i) {
            rB[i] = gDLoadOp(workspace[threadIdx.x % elems + i * elems]);
        }
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, workspace + bN)), sCLay);
        const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(Element, accumulator.data())),
            cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);

        // Reorder to striped arrangement
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            __syncthreads();
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = sC(rIdx + j, cIdx);
            }
        }

        // apply epilogue
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                rC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), rB[i]));
            }
        }

        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        // Coalesced copy from registers to global memory
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
            }
        }
    }

    // fused GEMM, epilogue and data transfer, with dynamic M and static N and K
    template<
        typename BlockGEMM,
        unsigned int N,
        unsigned int K,
        unsigned int elems = ACC::Elems::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void fGET(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& M,
        const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        static_assert(cute::size(accumulator) == bN);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::Stride<cute::Int<K>, cute::_1>{}));
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
            cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,
                cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
            make_layout(cute::make_shape(M, N), cute::Stride<cute::Int<N>, cute::_1>{}));

        // M is padded, such that the below is correct
        const auto tilesM = M / bM;
        // We assert the below prior to this point
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN),
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto k_tile_iter = cute::make_coord_iterator(tilesK);
        using Element = typename BlockGEMM::MatrixDType;
        // prefetch bias from global memory
        static_assert(ACC::sharedSize::value >= ACC::GSM::value + sizeof(Element) * bN);
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
            cute::Layout<cute::Shape<cute::_1, cute::Int<N>>,
                cute::Stride<cute::_0, cute::_1>>{});
        const auto biasCoord = idx2crd(tileIdx, cute::Shape<cute::_1, cute::Int<tilesN>>{},
            cute::Stride<cute::Int<bN>, cute::_1>{});
        const auto gD = cute::local_tile(mD,
            cute::Shape<cute::_1, cute::Int<bN>>{}, cute::get<1>(biasCoord));
        static_assert(threads % bN == 0);
        if (threadIdx.x < bN) {
            using LT =  cuda::std::conditional_t<sizeof(Element) == 2, uint16_t, uint32_t>;
            CAST_TO(LT, workspace)[threadIdx.x] = __ldg(CONST_CAST_TO(LT, &gD(threadIdx.x)));
        }
        using ElementC = typename decltype(accumulator)::value_type;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace + bN));
        /// There is a block-wide barrier at the end of the above ^

        // Epilogue
        typename BlockGEMM::MMA tiledMMA{};
        constexpr auto gCStoreOp = cutlass::NumericConverter<Element, ElementC>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<ElementC, Element>{};
        // Assume elementwise operator
        typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / elems;
        // copy single bias value
        ElementC rB[trips];
        #pragma unroll
        for (uint i = 0 ; i < trips; ++i) {
            rB[i] = gDLoadOp(workspace[threadIdx.x % elems + i * elems]);
        }
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, workspace + bN)), sCLay);
        const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(Element, accumulator.data())),
            cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);

        // Reorder to striped arrangement
        // TODO(Jonathan): Do the below reordering without shared memory. It would make my day (decade actually) to solve this.
        // A lot of performance badness happens down there.
        // I haven't sat down to think about a solution yet.
        // First idea that comes to mind is some form of warp register shuffling.
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            __syncthreads();
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = sC(rIdx + j, cIdx);
            }
        }

        // apply epilogue
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                rC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), rB[i]));
            }
        }

        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        // Coalesced copy from registers to global memory
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
            }
        }
    }

    // fused GEMM, epilogue and data transfer, with dynamic M and static N and K
    template<
        typename Operator,
        typename ActivationOp,
        unsigned int elems = 64
    >
    __forceinline__ __device__
    void fGETv2(typename Operator::ValueType* __restrict__ const& workspace,
        const typename Operator::ValueType* __restrict__ const& inputs,
        const typename Operator::ValueType* __restrict__ const& weights,
        typename Operator::ValueType* __restrict__ const& output,
        const typename Operator::ValueType* __restrict__ const& bias,
        const int& M,
        const int& tileIdx) {
        using BLAS = typename Operator::BLAS;
        static_assert(cublasdx::is_complete_blas_execution_v<BLAS>, "BLAS should be executable");
        using Element = typename BLAS::a_value_type;
        using ElementC = typename BLAS::c_value_type;
        constexpr auto gemmMainloop = typename Operator::Mainloop{};
        const auto partitioner = BLAS().suggest_partitioner();
        auto accumulator = partitioner.make_accumulator_fragment();
        cublasdx::clear(accumulator);
        using BM = typename Operator::BM;
        using BN = typename Operator::BN;
        using NT = typename Operator::NT;
        // Row-major
        const auto strideC = cute::conditional_return<
            cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major>
            (cute::Stride<NT, cute::_1>{}, cute::make_stride(cute::_1{}, M));
        const auto mC = cute::make_tensor(cute::make_gmem_ptr(output),
            cute::make_layout(cute::make_shape(M, NT{}), strideC));
        // M is padded, such that the below is correct
        const auto tilesM = M / BM{};
        using TilesN = cute::Int<NT{} / BN{}>;
        const auto tileCoord = cute::idx2crd(tileIdx, cute::Shape(tilesM, TilesN{}),
                cute::Stride<TilesN, cute::_1>{});
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gC = cute::local_tile(mC, cute::Shape<BM, BN>{}, cute::select<0, 1>(ctaCoord));
        // prefetch bias from global memory
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
            cute::Layout<cute::Shape<cute::_1, NT>, cute::Stride<cute::_0, cute::_1>>{});
        const auto biasCoord = idx2crd(tileIdx, cute::Shape<cute::_1, TilesN>{},
            cute::Stride<BN, cute::_1>{});
        const auto gD = cute::local_tile(mD, cute::Shape<cute::_1, BN>{}, cute::get<1>(biasCoord));
        if (threadIdx.x < BN{}) {
            using LT =  cuda::std::conditional_t<sizeof(Element) == 2, uint16_t, uint32_t>;
            CAST_TO(LT, workspace)[threadIdx.x] = __ldg(CONST_CAST_TO(LT, &gD(threadIdx.x)));
        }
        gemmMainloop(workspace + BN{}, accumulator, inputs, weights, output, M, tileIdx);
        // Epilogue
        constexpr auto gCStoreOp = cutlass::NumericConverter<Element, ElementC>{};
        constexpr auto epilogueOp = FAA<Element, ActivationOp>{};
        constexpr auto trips = cublasdx::size(accumulator) / elems;
        const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(Element, accumulator.data())),
            cute::Layout<cute::Shape<cute::_1, BN>, cute::Stride<BN, cute::_1>>{});
        cute::for_each(cute::make_int_sequence<cublasdx::size(accumulator)>{}, [&](auto j) {
            rC(j) = gCStoreOp(accumulator(j));
        });
        // copy single bias value
        Element rB[trips];
        #pragma unroll
        for (uint i = 0 ; i < trips; ++i) {
            rB[i] = workspace[threadIdx.x % elems + i * elems];
        }
        __syncthreads();
        auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(Element, workspace)),
            cute::Layout<cute::Shape<BM, BN>,
            cuda::std::conditional_t<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major,
                cute::Stride<BN, cute::_1>, cute::Stride<cute::_1, BM>>>{});
        // rmem -> smem
        cublasdx::copy_fragment<cublasdx::alignment_of_v_c<BLAS>>(accumulator, sC, partitioner);
        __syncthreads();
        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        // smem -> rmem
        cute::for_each(cute::make_int_sequence<trips>{}, [&](auto i) {
            cute::for_each(cute::make_int_sequence<elems>{}, [&](auto j) {
                rC(j + i * elems) = sC(rIdx + j, cIdx);
            });
        });
        // epilogue
        cute::for_each(cute::make_int_sequence<trips>{}, [&](auto i) {
            cute::for_each(cute::make_int_sequence<elems>{}, [&](auto j) {
                rC(j + i * elems) = epilogueOp(rC(j + i * elems), rB[i]);
            });
        });
        // rmem -> gmem
        cute::for_each(cute::make_int_sequence<trips>{}, [&](auto i) {
            cute::for_each(cute::make_int_sequence<elems>{}, [&](auto j) {
                gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
            });
        });
    }

    struct __align__(16) ProcessorArgs{
        // sensible sentinel values
        unsigned int* __restrict__ sQ = nullptr;
        TQSignal* __restrict__ pDB = nullptr;
        unsigned int* __restrict__ tQH = nullptr;
        Task* __restrict__ tQ = nullptr;
        Task* __restrict__ ptQ = nullptr;
        unsigned int* __restrict__ tQS = nullptr;

        ProcessorArgs() = default;
        __device__
        ProcessorArgs(unsigned int* const& _sQ,
            TQSignal* const& _pDB,
            unsigned int* const& _tQH,
            Task* const& _tQ,
            Task* const& _ptQ,
            unsigned int* const& _tQS) :
        sQ(_sQ), pDB(_pDB), tQH(_tQH), tQ(_tQ), ptQ(_ptQ), tQS(_tQS) {}
    };

    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value
    >
    __device__ __forceinline__
    void notifyNext(uint* __restrict__ const& workspace, const Task& rCurrentTask, const ProcessorArgs& pA) {
        static_assert(sizeof(Task) == 128);
        constexpr auto eS = sizeof(Task) / sizeof(uint);
        static_assert(eS == WARP_SIZE);
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        static_assert(sharedSize % sizeof(Task) == 0);
        static_assert(sharedSize / sizeof(Task) >= threads);
        constexpr auto capacity = threads;
        constexpr auto trips = tasks / capacity;
        static_assert(threads % eS == 0);
        static_assert(capacity % threads == 0);
        constexpr auto elems = capacity * eS / threads;
        constexpr unsigned int preIndex = 0;

        const auto offset = ACC::TNx::value * rCurrentTask.batchIdx;
        auto* __restrict__ tQ = CAST_TO(uint, pA.ptQ + (rCurrentTask.syncIdx * ACC::TNx::value));
        const auto cIdx = threadIdx.x % eS;
        // prep memory-view tensors
        const auto sTQ = make_tensor(cute::make_smem_ptr(workspace),
            cute::Layout<cute::Shape<cute::Int<threads>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        const auto gTQ = make_tensor(cute::make_gmem_ptr(tQ),
            cute::Layout<cute::Shape<cute::Int<tasks>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        // copy from registers to shared memory using swizzle
        if constexpr (trips) {
            const auto rIdx = threadIdx.x / eS * eS;
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // each thread does a copy from registers to shared memory
                const auto taskIdx = threadIdx.x + i * capacity;
                const auto tileIdx = offset + taskIdx;
                const auto nextTask = Task {
                    TaskType::postGEMM,
                    rCurrentTask.cData[preIndex],
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + offset + (p == PeerConnectivity::p2p ? taskIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                };
                // Directive to the compiler to reinterpret the Task structure as a stream of 4-byte blocks
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    // temporal shift of indices to eliminate bank conflicts
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
                __syncthreads();
                // now copy from shared memory to global memory
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    gTQ(rIdx + (j + i * capacity), cIdx) = sTQ(rIdx + j, (cIdx + j) % eS);
                }
            }
            // before reusing shared memory below
            __syncthreads();
        }
        if constexpr (constexpr auto residue = tasks - trips * capacity; residue) {
            if (threadIdx.x < residue) {
                const auto taskIdx = threadIdx.x + trips * capacity;
                const auto tileIdx = offset + taskIdx;
                const auto nextTask = Task {
                    TaskType::postGEMM,
                    rCurrentTask.cData[preIndex],
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + offset + (p == PeerConnectivity::p2p ? taskIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                };
                // Directive to the compiler to reinterpret the Task structure as a stream of 4-byte blocks
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    // temporal shift of indices to eliminate bank conflicts
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
            }
            __syncthreads();
            constexpr auto stride = threads / eS;
            const auto pIdx = threadIdx.x / eS;
            constexpr auto length = residue / stride;
            // now copy from shared memory to global memory by multiplexing each row across available warps
            #pragma unroll
            for (uint j = 0; j < length; ++j) {
                const auto idx = j * stride + pIdx;
                gTQ(idx + trips * capacity, cIdx) = sTQ(idx, (cIdx + idx) % eS);
            }
            if constexpr (constexpr auto rS = residue % stride; rS) {
                if (pIdx < rS) {
                    const auto idx = length * stride + pIdx;
                    gTQ(idx + trips * capacity, cIdx) = sTQ(idx, (cIdx + idx) % eS);
                }
            }
        }

        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            // notify scheduler
            atomicAdd(pA.tQH + rCurrentTask.syncIdx, tasks);
        }
    }

    template<
        typename ScaleWeights,
        typename Output
    >
    __device__ __forceinline__
    void start(cuda::std::byte* const& workspace,
        ScaleWeights const& sW, Output const& moeOutput,
        const uint16_t& _seqBit){
        using Element = ACC::Element;
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(flagsType)
            && alignof(SignalPayload<PacketStage::last>) == alignof(flagsType));

        static_assert(sizeof(Task) == 128);
        __shared__ Task currentTask;
        __shared__ uint globalInterrupt;
        __shared__ uint enqueue;

        // Register allocations
        const auto rSeqBit = _seqBit;
        Task rCurrentTask{};
        TQSignal tqs{0U, 0U};
        const auto pA = ProcessorArgs{
            bookkeeping.sQ() + blockIdx.x,
            bookkeeping.pDB() + blockIdx.x,
            bookkeeping.tQH(),
            bookkeeping.tQ(),
            bookkeeping.ptQ(),
            bookkeeping.tSA()
        };

        if (!threadIdx.x) {
            atomicExch_block(&globalInterrupt, 0U);
            atomicExch_block(&enqueue, 0U);
        }
        using PreGEMM = BlockMM<ACC::ActivationOp, Element>;
        using PostGEMM = BlockMM<ACC::ActivationOpX, Element>;
        using PRO = GO<ACC::P::value, ACC::H::value>;
        using PSO = GO<ACC::H::value, ACC::P::value>;
        constexpr uint H = ACC::H::value;
        constexpr auto tN = ACC::TN::value;
        constexpr auto tNx = ACC::TNx::value;
        __syncthreads();
        while (!tqs.interrupt) {
            if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                if (!threadIdx.x) {
                    auto* __restrict__ tQSignal = pA.pDB;
                    // Grabs next task
                    awaitNotification(tQSignal, &tqs, tqs.signal);
                    __threadfence();
                    // Eagerly indicate readiness for the next task as the above fence allows us to do so correctly
                    globalInterrupt = tqs.interrupt;
                    atomicExch(pA.sQ, ready);
                }
                // The below is necessary as it guarantees memory ordering
                __syncwarp();
                auto* __restrict__ tqsP = CAST_TO(ull_t, &tqs);
                *tqsP = __shfl_sync(0xffffffff, *tqsP, 0);
                const auto* __restrict__ gtQ = pA.tQ + tqs.decodeSig();
                if (!tqs.interrupt) {
                    // coalesced copy from global to shared memory
                    CAST_TO(uint, &currentTask)[threadIdx.x] = __ldg(CONST_CAST_TO(uint, gtQ) + threadIdx.x);
                }
            }
            __syncthreads();
            tqs.interrupt = globalInterrupt;
            // if we receive an interrupt, there is nothing to do next
            if (!tqs.interrupt) {
                // shared -> registers
                rCurrentTask = currentTask;
                switch (rCurrentTask.taskType) {
                    case TaskType::preGEMM: {
                        constexpr unsigned int preIndex = 0;
                        fGETv2<PRO, AFunction<HIDDEN_ACT, Element>::DT>(
                            CAST_TO(typename PRO::ValueType, workspace),
                            CONST_CAST_TO(typename PRO::ValueType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PRO::ValueType, rCurrentTask.bData[preIndex]),
                            CAST_TO(typename PRO::ValueType, rCurrentTask.cData[preIndex]),
                            CONST_CAST_TO(typename PRO::ValueType, rCurrentTask.dData[preIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tN;
                        }
                        __syncthreads();
                        if (enqueue) {
                            if (!rCurrentTask.isPeerRemote) {
                                notifyNext<PeerConnectivity::p2p>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                            else {
                                notifyNext<PeerConnectivity::remote>(CAST_TO(uint, workspace), rCurrentTask, pA);
                            }
                        }
                    }
                    break;
                    case TaskType::postGEMM: {
                        constexpr unsigned int postIndex = 1;
                        fGETv2<PSO, cublasdx::identity>(
                            CAST_TO(typename PRO::ValueType, workspace),
                            CONST_CAST_TO(typename PRO::ValueType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PRO::ValueType, rCurrentTask.bData[postIndex]),
                            CAST_TO(typename PRO::ValueType, rCurrentTask.cData[postIndex]),
                            CONST_CAST_TO(typename PRO::ValueType, rCurrentTask.dData[postIndex]),
                            rCurrentTask.M,
                            currentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            // Pack payload into single signal word of 8 bytes
                            const auto flagSignal = SignalPayload<PacketStage::last>{
                                rCurrentTask.batchIdx,
                                rCurrentTask.tileSize,
                                rSeqBit,
                            };
                            if (rCurrentTask.isPeerRemote) {
                                // Remote; check if we need to do the transfer
                                __threadfence();
                                if (atomicIncrement(pA.tQS + rCurrentTask.syncIdx) + 1 == tN + tNx) {
                                    nvshmem_putmem_signal_nbi(rCurrentTask.rcData,
                                        rCurrentTask.cData[postIndex],
                                        // Batched remote network transfer to avoid overwhelming the NIC
                                        rCurrentTask.tileSize * H * sizeof(Element),
                                        rCurrentTask.flags,
                                        *CONST_CAST_TO(flagsType, &flagSignal), NVSHMEM_SIGNAL_SET,
                                        rCurrentTask.peerIdx);
                                }
                            }
                            else {
                                // individual tile, no batching here
                                // Already did the network transfer,
                                // so set signal only
                                __threadfence_system();
                                atomicExch_system(CAST_TO(ull_t, rCurrentTask.flags),
                                    *CONST_CAST_TO(ull_t, &flagSignal));
                            }
                        }
                    }
                    break;
                    case TaskType::combine: {
                        constexpr unsigned int combineIndex = 0;
                        combine<ACC::CM::value>(
                            workspace,
                            CONST_CAST_TO(TPS, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.bData[combineIndex]),
                            moeOutput.data().get(),
                            sW,
                            rCurrentTask.tileIdx,
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
                    }
                    break;
                }
            }
        }
    }
}
#endif //KLEOS_COMPUTE_CUH
