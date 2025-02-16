# cp.async
```
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::pref}                 
                         [dst], [src], cp-size{, src-size}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, src-size}{, cache-policy} ;
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, ignore-src}{, cache-policy} ;

.level::cache_hint =     { .L2::cache_hint }
.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }
cp-size =                { 4, 8, 16 }
```
## experient
### .level::prefetch_size
32 * 4B L2::128B
tid0~15 cp.async 16 * 4B = 64B
tid0 load 第64B即a[17]
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test(int* a) {
    int tid = threadIdx.x;

    __shared__ int smem[32];
    unsigned int asl = __cvta_generic_to_shared(smem + threadIdx.x);
    int src_size = 4;
    if (tid >= 0 && tid <= 15) {// 64B 128B 
        asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2, %3;\n" ::"r"(asl),
                     "l"(a + tid),
                     "n"(sizeof(int)),
                     "r"(src_size));
    }

    asm volatile("cp.async.wait_all;\n" ::);
    if (tid == 0) {
        int data = a[17];
    }
}
int main() {
    int* a = new int[64];
    for (int i = 0; i < 64; i++) {
        a[i] = i + (1 << 8) + (1 << 16);
    }
    int* da;
    cudaMalloc(&da, sizeof(int) * 32);
    cudaMemcpy(da, a, sizeof(int) * 32, cudaMemcpyHostToDevice);
    test<<<1, 32>>>(da);
    cudaDeviceSynchronize();
}
```
### cp-size{, src-size}
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test(int* a) {
    int tid = threadIdx.x;

    __shared__ int smem[32];
    unsigned int asl = __cvta_generic_to_shared(smem + threadIdx.x);
    int src_size = 4;//4,2,1观察现象
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" 
                ::"r"(asl),
                "l"(a + tid),
                "n"(sizeof(int)),
                "r"(src_size));

    asm volatile("cp.async.wait_all;\n" ::);

    if (tid == 0) {
        for (int i = 0; i < 32; i++) {
            printf("%d ", smem[i]);
        }
    }
}
int main() {
    int* a = new int[32];
    for (int i = 0; i < 32; i++) {
        a[i] = i + (1 << 8) + (1 << 16);
    }
    int* da;
    cudaMalloc(&da, sizeof(int) * 32);
    cudaMemcpy(da, a, sizeof(int) * 32, cudaMemcpyHostToDevice);
    test<<<1, 32>>>(da);
    cudaDeviceSynchronize();
}
   
```
### 16{, ignore-src}
### bankconflict?