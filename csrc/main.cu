/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include <cublasdx.hpp>
/*#include "include/kleos/bootstrap.cuh"
#include "include/kleos/moe/moe.cuh"

__host__ __forceinline__
void runOS() {
    kleos::initialize();
    const auto rank = kleos::getRank();
    // generate random input tile and eye weights
    constexpr auto S = kleos::ACC::S::value;
    constexpr auto H = kleos::ACC::H::value;
    constexpr auto E = kleos::ACC::E::value;
    constexpr auto P = kleos::ACC::P::value;
    constexpr auto PX = kleos::ACC::PX::value;
    const auto nLx = kleos::hostBookkeeping.nLx;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    const auto bZ =  gwZ + nLx * P * H;
    const auto b2Z =  bZ + nLx * P * H;
    const auto dZ =  b2Z + nLx * (P + H);
    const auto gZ = dZ + S * PX;
    const auto cZ = gZ + S * H;
    cuda::std::byte* p;
    KLEOS_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), kleos::kleosStream));
    KLEOS_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), kleos::kleosStream));
    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    using Element = kleos::ACC::Element;
    auto* __restrict__ eHp = static_cast<Element*>(hP);
    {
        #if KLEOS_NVTX
        kleos::kleosRange forwardRange{"Host Data Prep"};
        #endif
        thrust::default_random_engine rng(47 * (rank + 42));
        thrust::normal_distribution<float> dist(0, 5);
        // Activations
        thrust::generate(fHp, fHp + aZ, [&] { return dist(rng); });
        // gate weights
        thrust::generate(fHp + aZ, fHp + aZ + E * H, [&] { return dist(rng); });
        // Expert weights
        // loop for number of experts
        for (uint i = 0; i < nLx; ++i) {
            // expert up
            thrust::generate(fHp + gwZ + i * (P * H), fHp + gwZ + (i + 1) * (P * H),
                [&] { return dist(rng); });
            thrust::generate(fHp + bZ + i * (P * H), fHp + bZ + (i + 1) * (P * H),
                [&] { return dist(rng); });
        }
        // bias
        std::ranges::fill(fHp + b2Z, fHp + dZ, 0.0f);
        constexpr cutlass::NumericConverter<Element, float> conv{};
        for (uint i = 0; i < dZ; ++i) {
            eHp[i] = conv(fHp[i]);
        }
    }
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ,
        cudaMemcpyHostToDevice,
        kleos::kleosStream));
    float timed = 0;
    kleos::moe::forwardHostBench<32, 32>(p, p + dZ * sizeof(Element), timed);
    printf("epRank: %u took %.2fms\n", kleos::hostBookkeeping.rank, timed);
    KLEOS_CHECK_CUDA(cudaPeekAtLastError());
    kleos::finalize();
    std::free(hP);
}*/
#include "include/kleos/types.cuh"
#include "include/kleos/debug.cuh"
#include "include/kleos/os/processor/gemm.cuh"

template<
    cublasdx::arrangement ta,
    cublasdx::arrangement tb,
    cublasdx::arrangement tc,
    int threads, int AlignmentBytes,
    int tM, int tN, int tK,
    int N, int K, int pipeStages,
    typename ElementC,
    typename ElementA,
    typename ElementB,
    typename ElementOut
>
__global__ void play(const ElementA* __restrict__ a,
            const ElementB* __restrict__ b, ElementOut* __restrict__ c, const int M) {
    using BLAS = decltype(
        cublasdx::Size<tM, tN, tK>() +
        cublasdx::Precision<ElementA, ElementB, ElementC>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<ta, tb, tc>() +
        cublasdx::Block() +
        cublasdx::BlockDim<threads>() +
        cublasdx::Alignment<AlignmentBytes, AlignmentBytes, AlignmentBytes>() +
        cublasdx::experimental::StaticBlockDim() +
        cublasdx::SM<KLEOS_ARCH>());
    __shared__ __align__(AlignmentBytes) cuda::std::byte workspace[kleos::getSharedSize<BLAS, pipeStages, AlignmentBytes>()];
    constexpr auto gemmMainloop = kleos::GM<BLAS, N, K, pipeStages>{};
    const auto partitioner = BLAS().suggest_partitioner();
    auto accumulator = partitioner.make_accumulator_fragment();
    using BM = cute::Int<tM>;
    using BN = cute::Int<tN>;
    using BT = cute::Shape<cute::Int<tM>, cute::Int<tN>, cute::Int<tK>>;
    using NT = cute::Int<N>;
    const auto strideC = cute::conditional_return<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major>
            (cute::Stride<NT, cute::_1>{}, cute::make_stride(cute::_1{}, M));
    const auto mC = cute::make_tensor(cute::make_gmem_ptr(c),
        cute::make_layout(cute::make_shape(M, N), strideC));
    const auto tilesM = M / BM{};
    using tilesN = cute::Int<(N / BN{})>;
    const auto nTiles = tilesM * tilesN{};
    auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementOut, workspace)),
        cute::Layout<cute::Shape<BM, BN>,
            cute::conditional_t<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major,
                cute::Stride<BN, cute::_1>, cute::Stride<cute::_1, BM>>>{});
    for (auto tileIdx = blockIdx.x; tileIdx < nTiles; tileIdx += blockDim.x) {
        cublasdx::clear(accumulator);
        const auto tileCoord = cute::idx2crd(tileIdx, cute::Shape(tilesM, tilesN{}),
                cute::Stride<cute::Int<tilesN{}>, cute::_1>{});
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        auto gC = cute::local_tile(mC, cute::Shape<BM, BN>{}, cute::select<0, 1>(ctaCoord));
        gemmMainloop(workspace, accumulator, a, b, c, M, tileIdx);
        cublasdx::copy_fragment<cublasdx::alignment_of_v_c<BLAS>>(accumulator, gC, partitioner);
        cublasdx::copy_fragment<cublasdx::alignment_of_v_c<BLAS>>(accumulator, sC, partitioner);
    }
}
int main() {
    using Element = __half;
    using ElementCompute = float;
    constexpr auto M  = 32;
    constexpr auto N  = 32;
    constexpr auto K  = 32;

    constexpr auto tM  = 32;
    constexpr auto tN  = 32;
    constexpr auto tK  = 32;
    constexpr auto threads = 64;
    constexpr auto pipeStages = cute::min(2, K / tK);
    Element* dP;

    KLEOS_CHECK_CUDA(cudaMallocManaged(&dP, (K * (M + N) + M * N) * sizeof(Element)));
    KLEOS_CHECK_CUDA(cudaMemset(dP, 0, (K * (M + N) + M * N) * sizeof(Element)));

    Element* dA = dP;
    Element* dB = dP + (M * K);
    Element* dC = dP + (K * N);

    // fill
    dA[0] = 0.f;
    dA[1] = 1.f;
    dA[0 + K] = 2.f;
    dA[1 + K] = 3.f;

    dB[0] = 4.f;
    dB[1] = 5.f;
    dB[0 + K] = 6.f;
    dB[1 + K] = 7.f;

    constexpr auto ta = cublasdx::row_major;
    constexpr auto tb = cublasdx::col_major;
    constexpr auto tc = cublasdx::row_major;

    using MT = cute::Int<K>;
    using KT = cute::Int<K>;
    using NT = cute::Int<N>;

    constexpr auto AlignmentBytes = 16;

    using BLAS = decltype(
        cublasdx::Size<128, 128, 32>() +
        cublasdx::Precision<Element, Element, ElementCompute>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::MM>() +
        cublasdx::Arrangement<ta, tb, tc>() +
        cublasdx::Block() +
        cublasdx::BlockDim<threads>() +
        cublasdx::Alignment<AlignmentBytes, AlignmentBytes, AlignmentBytes>() +
        cublasdx::experimental::StaticBlockDim() +
        cublasdx::SM<KLEOS_ARCH>());

    constexpr auto z = kleos::getSharedSize<BLAS, 2, 16>();
    printf("z is %d\n", z);

    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    using BK = cute::Int<cublasdx::size_of<BLAS>::k>;
    constexpr auto sALay = cute::tile_to_shape(BLAS::suggest_layout_smem_a().layout,
                cute::Shape<BM, BK, cute::Int<pipeStages>>{});
    constexpr auto sBLay = cute::tile_to_shape(BLAS::suggest_layout_smem_b().layout,
        cute::Shape<BK, BN, cute::Int<pipeStages>>{});

    /*using strideA = cuda::std::conditional_t<ta == cublasdx::row_major,
            cute::Stride<KT, cute::_1>, cute::Stride<cute::_1, MT>>;
    const auto tA = cute::make_tensor(CAST_TO(cute::half_t, dA), cute::Layout<cute::Shape<MT, KT>, strideA>{});
    print_tensor(tA);
    using strideB = cuda::std::conditional_t<tb == cublasdx::row_major,
            cute::Stride<NT, cute::_1>, cute::Stride<cute::_1, KT>>;
    const auto tB = cute::make_tensor(CAST_TO(cute::half_t, dB), cute::Layout<cute::Shape<KT, NT>, strideB>{});
    print_tensor(tB);*/
    // call GEMM
    play<ta, tb, tc, threads, AlignmentBytes, tM, tN, tK, N, K, pipeStages, ElementCompute><<<1, threads>>>(dA, dB, dC, M);
    KLEOS_CHECK_CUDA(cudaDeviceSynchronize());
    /*using strideC = cuda::std::conditional_t<tc == cublasdx::row_major,
            cute::Stride<NT, cute::_1>, cute::Stride<cute::_1, MT>>;
    const auto tC = cute::make_tensor(CAST_TO(cute::half_t, dC), cute::make_layout(cute::make_shape(M, N), strideC{}));
    print_tensor(tC);*/
    KLEOS_CHECK_CUDA(cudaFree(dP));
    //runOS();
}
