# block thread
以v100为例
目标是提高occupancy

## thread
max warps per sm 64
max blocks per sm 32
max threads per sm 2048
max threads per block 1024
这组数据可以计算出一个block的最小threads 2048/32=64，即当一个block的threads小于64，其无论如何占用率也无法达到100%
且实际threads要是max threads的约数,64,128,256,512...

## constrain
- max shared mem per sm && max shared mem per block
- max register per sm && max register per block && max register per thread 

给定实际单个thread使用寄存器r，实际shared mem大小s，实际thread为t,计算实际占用率
1. r < max register per thread 
2. s < max shared mem per block
3. t < max threads per block

reg_limit = max register per sm / (t * r)
smem_limit = max smem per sm / s
sm_limit = max threads per sm / t

block = min(reg_limit,smem_limit,sm_limit)
occupancy = block * t / max threads per sm