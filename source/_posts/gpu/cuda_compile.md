---
title: cuda compile
date: 2024-06-12 
tags: gpu
---
cuda的编译流程
<!--more-->
## NVCC编译流程
1. 编译过程，device code单独编译，host code包含device code接口文件.cudafe1.stub.c，之后正常gcc编译生成目标文件
2. 链接过程，device code 单独（seperate）链接，之后再和上步的目标文件链接生成可执行文件
![](./nvcc.png)
下面为真实编译流程,删除了部分信息
```shell
 gcc -E -x c++ "warp.cu" -o "warp.cpp4.ii" 
 cudafe++ --c++14 --orig_src_file_name "warp.cu" --gen_c_file_name "warp.cudafe1.cpp" --stub_file_name "warp.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "warp.module_id" "warp.cpp4.ii" 

 gcc -E -x c++ "warp.cu" -o "warp.cpp1.ii" 
 cicc --c++14 --orig_src_file_name "warp.cu" --include_file_name "warp.fatbin.c" --module_id_file_name "warp.module_id" --gen_c_file_name "warp.cudafe1.c" --stub_file_name "warp.cudafe1.stub.c" --gen_device_file_name "warp.cudafe1.gpu"  "warp.cpp1.ii" -o "warp.ptx"
 ptxas -arch=sm_52 -m64  "warp.ptx"  -o "warp.sm_52.cubin" 
 fatbinary --create="warp.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=52,file=warp.sm_52.cubin" "--image3=kind=ptx,sm=52,file=warp.ptx" --embedded-fatbin="warp.fatbin.c" 

 gcc -c -x c++ "warp.cudafe1.cpp" -o "warp.o" 

 # 链接过程
 nvlink -m64 --arch=sm_52 --register-link-binaries="test_dlink.reg.c"    "-L/home/xiongqian/cuda-12.4/bin/../targets/x86_64-linux/lib/stubs" "-L/home/xiongqian/cuda-12.4/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "warp.o"  -lcudadevrt  -o "test_dlink.sm_52.cubin" --host-ccbin "gcc"
 fatbinary --create="test_dlink.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=52,file=test_dlink.sm_52.cubin" --embedded-fatbin="test_dlink.fatbin.c" 
 gcc -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -c -x c++ -DFATBINFILE="\"test_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"test_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -Wno-psabi "-I/home/xiongqian/cuda-12.4/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=131 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/home/xiongqian/cuda-12.4/bin/crt/link.stub" -o "test_dlink.o" 
 g++ -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -m64 -Wl,--start-group "test_dlink.o" "warp.o"   "-L/home/xiongqian/cuda-12.4/bin/../targets/x86_64-linux/lib/stubs" "-L/home/xiongqian/cuda-12.4/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "test"
```

## separate compilation
![](./nvcc-options-for-separate-compilation.png)
```
// device code指的是CUDA相关代码，host object指的是c++代码编译出来的产物
// 将x.cu和y.cu中的device code分别嵌入到其对应的host object中，即x.o和y.o
nvcc --gpu-architecture=sm_50 --device-c x.cu y.cu
// 使用device-link将x.o和y.o中的device code链接在一起，得到link.o
nvcc --gpu-architecture=sm_50 --device-link x.o y.o --output-file a_dlink.o
// 将链接后的link.o和其他host object链接在一起，得到最终产物
g++ x.o y.o a_dlink.o -L<path> -lcudart // 这里<path>替换成你libcudart.so对应路径
```

## virtual and real arch and just intime compilation
![](./nvcc-jit.png)
`-arch=compute_70`指定虚拟架构，stage1生成ptx文件
`-code=compute_70,sm_70`根据虚拟架构生成cubin文件，并指定哪些文件打包进fatbin
具体看下面的参考blog

## command option
### compilation
`-link`Specify the **default** behavior: compile and link all input files.
`-c`.cu -> .o
> 以上默认不开启Separate Compilation,因此使用extern关键字，多文件链接会报错

`--device-c`&`-dc`启用Separate Compilation .cu->.o
`--device-link`&`-dlink`启用Separate Compilation链接
`--relocatable-device-code=true --compile`合并上面两个过程

`-cuda`是对.cudafe1.cpp的预处理

`-ptx`&`-cubin`舍弃host code，仅包含device code，其中-cubin二进制文件反汇编即为sass代码
### nvcc driver

## cuobjdump
`cuobjdump -all`
`cuobjdump -lptx/-lelf`
`cuobjdump -ptx/-sass`

## nvdisasm

## 参考
[cuda nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#)
[cuobjdump nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities/#)
[blog 1](https://wangpengcheng.github.io/2019/04/17/nvcc_learn_note/)
[blog 2](https://polobymulberry.github.io/2019/03/04/CUDA%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97%281%29%20%7C%20%E7%BC%96%E8%AF%91%E9%93%BE%E6%8E%A5%E7%AF%87/)
[blog 3](https://www.zhangty15226.com/2023/11/25/NVCC%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B/#nvcc-%E7%BC%96%E8%AF%91%E6%B5%81%E7%A8%8B)
[blog 4](https://zhengqm.github.io/blog/2018/12/07/cuda-nvcc-tips.html)

