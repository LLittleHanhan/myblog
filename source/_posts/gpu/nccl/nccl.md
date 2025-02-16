# nccl
[document](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduce)
## demo
```cpp
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
using namespace std;


int main() {
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    // Initialize NCCL
    ncclComm_t comms[num_devices];
    cudaStream_t streams[num_devices];

    // Allocate buffer for data on each device
    float* data[num_devices];
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc(&data[i], sizeof(float));
        cudaStreamCreate(&streams[i]);
    }

    // Initialize NCCL communicators for each device
    int* devs = new int[num_devices];
    for (int i = 0; i < num_devices; i++) {
        devs[i] = i;
    }
    NCCL_CHECK(ncclCommInitAll(comms, num_devices, devs));

    // Set the data (just an example)
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        float value = (float)i;
        cudaMemcpyAsync(data[i], &value, sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // Perform All-Reduce
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        ncclAllReduce((const void*)data[i], (void*)data[i], 1, ncclFloat, ncclSum, comms[i], streams[i]);
    }


    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }


    for (int i = 0; i < num_devices; i++) {
        float value;
        cudaMemcpy(&value, data[i], sizeof(float), cudaMemcpyDeviceToHost);
        cout << value << endl;
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
        cudaFree(data[i]);
    }

    delete[] devs;
    return 0;
}

```
## 
nccl用于gpu通信
nccl的通信原语
- p2p send recive
- p2a gather scatter
- a2a reduce broadcast allreduce allgather reducescatter
  
## allreduce
1. reduce + broadcast
2. reducescatter + all gather 