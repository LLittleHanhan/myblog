# cute detail

## mma and ldsm
A * B = C
这里讨论A和B的主序问题
- 首先对于mma指令来说，它的划分是不需要考虑主序问题，唯一需要注意的是对于cute来说，B矩阵是需要表示成转置后的形式
- 麻烦的是ldsm指令，它需要考虑主序问题以实现拷贝正确的数字，这里矩阵的形状同mma，对于不同的主序使用不同的ldsm指令
### example
```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void f() {
    __shared__ half A[32 * 16];  // row_major
    __shared__ half B[16 * 32];  // row_major
    __shared__ half C[32 * 32];
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 16; j++) {
                A[i * 16 + j] = __float2half(i * 16 + j);
            }
        }
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 32; j++) {
                B[i * 32 + j] = __float2half(i * 32 + j);
            }
        }
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%.1f ", __half2float(A[i * 16 + j]));
            }
            printf("\n");
        }
        printf("\n\n\n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 32; j++) {
                printf("%.1f ", __half2float(B[i * 32 + j]));
            }
            printf("\n");
        }
    }
    __syncthreads();
    
    
    using MMA = decltype(make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        Layout<Shape<_2, _2, _1>>{},
        Tile<_32, Layout<Shape<_32>, Stride<_1>>, Layout<Shape<_16>, Stride<_1>>>{}));

    
    auto sA = make_tensor(make_smem_ptr(A), Shape<_32, _16>{}, Stride<_16, _1>{});
    // auto sB = make_tensor(make_smem_ptr(B), Shape<_16, _32>{}, Stride<_32, _1>{});
    
    // sB row major
    auto sB_R = make_tensor(make_smem_ptr(B), Shape<_16, _32>{}, Stride<_32, _1>{});
    auto sB_R_T = make_tensor(make_smem_ptr(B), Shape<_32, _16>{}, Stride<_1,_32>{});
    // sB col major
    auto sB_C = make_tensor(make_smem_ptr(B), Shape<_16, _32>{}, Stride<_1, _16>{});
    auto sB_C_T = make_tensor(make_smem_ptr(B), Shape<_32, _16>{}, Stride<_16,_1>{});
    
    // auto sB_X = make_tensor(make_smem_ptr(B), Shape<_32, _16>{}, Stride<_32, _1>{});
    auto sC = make_tensor(make_smem_ptr(C), Shape<_32, _32>{}, Stride<_32, _1>{});

    MMA mma;
    auto thr_mma = mma.get_slice(tid);
    auto mma_tArA = thr_mma.partition_fragment_A(sA);
    auto mma_tBrB = thr_mma.partition_fragment_B(sB_C_T);
    auto mma_tCrC = thr_mma.partition_fragment_C(sC);
    if (tid == 0) {
        printf("mma_tArA \n");
        print(mma_tArA);
        printf("\n");
        printf("mma_tBrB \n");
        print(mma_tBrB);
        printf("\n");
        printf("mma_tCrC \n");
        print(mma_tCrC);
        printf("\n");
    }

    using S2RCopyB = decltype(make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, MMA{}));
    using S2RCopyBT = decltype(make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, MMA{}));
    S2RCopyBT s2r_copy_B;
    auto  s2r_thr_copy_B = s2r_copy_B.get_slice(tid);
    auto s2r_tBsB = s2r_thr_copy_B.partition_S(sB_R);
    auto s2r_tBrB = s2r_thr_copy_B.retile_D(mma_tBrB_T);
    copy(s2r_copy_B,s2r_tBsB,s2r_tBrB);
    
    __syncthreads();
    if (tid == 0) {
        printf("s2r_tBrB \n");
        print_tensor(s2r_tBrB);
        printf("\n");
    }

    // __syncthreads();
    // gemm(mma, mma_tCrC,mma_tArA,mma_tBrB,mma_tCrC);
}

int main() {
    f<<<1, 128>>>();
    cudaDeviceSynchronize();
}
```
只需要考虑sB_R_T和sB_C_T两种tensor，前者是行主序，后者是列主序
对于mma指令来说，对B_T的取数是按行
因此三sB_R_T原始是行主序，实际转置后是列主序，和mma相反，使用S2RCopyBT