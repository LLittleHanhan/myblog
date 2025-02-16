# tencore pipeline utilization分析

## 16 * 8 * 16
### 一
循环内只有一次mma指令
```cpp
asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13};\n"
            : "=f"(r1[0]), "=f"(r1[1]), "=f"(r1[2]), "=f"(r1[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r1[0]), "f"(r1[1]), "f"(r1[2]), "f"(r1[3]));
```

sass
```
        /*0200*/                   HMMA.16816.F32 R4, R8, R2, R4 ;            /* 0x000000020804723c */
                                                                              /* 0x020f5e0000001804 */
        /*0210*/                   NOP ;                                      /* 0x0000000000007918 */
                                                                              /* 0x000fd20000000000 */
        /*0220*/                   HMMA.16816.F32 R4, R8, R2, R4 ;            /* 0x000000020804723c */
                                                                              /* 0x020f5e0000001804 */
        /*0230*/                   NOP ;                                      /* 0x0000000000007918 */
                                                                              /* 0x000fd20000000000 */
        /*0240*/                   HMMA.16816.F32 R4, R8, R2, R4 ;            /* 0x000000020804723c */
                                                                              /* 0x020f5e0000001804 */
        /*0250*/                   NOP ;                                      /* 0x0000000000007918 */
```
- mma 020f5e 为 15 cycle
- nop 000fd2 为 9 cycle

ncu b 128 t 128 mma 1000
```
Elapsed Cycles                cycle        37921
SM Active Cycles              cycle     34899.31
SM Frequency                    Ghz         2.23

Average SM Active Cycles         cycle     34899.31
Total SM Elapsed Cycles          cycle      4857326
Average SMSP Active Cycles       cycle     34920.54
Total SMSP Elapsed Cycles        cycle     19429304

smsp__pipe_tensor_op_hmma_cycles_active_v2.avg                                         cycle        16000
smsp__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active                %        45.82
smsp__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_elapsed               %        42.16

smsp__pipe_tensor_op_hmma_cycles_active_v2.avg.peak_sustained_active                   cycle     34920.54
smsp__pipe_tensor_op_hmma_cycles_active_v2.avg.peak_sustained_elapsed                  cycle     37947.86
```

4090
- 512 tensorcores 
- 128 flop per cycle
- 2.52 cycle/ns
- tensor fp16 算力 165.2 * 10^12 flops

Q:
1. 为什么smsp__pipe_tensor_op_hmma_cycles_active_v2是16000？ 1000*16，为什么是16而不是32
   16 * 16 * 8 * 2 / 128 = 32
   throughput: 16 * 16 * 8 * 2 * 1000 / 34920.54 * 128  = 91.63%
   或 1000 * 32 / 34920.54 = 91.63%

### 二
一中有数据依赖，循环内发射多个mma
两个
```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(r1[0]), "=f"(r1[1]), "=f"(r1[2]), "=f"(r1[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r1[0]), "f"(r1[1]), "f"(r1[2]), "f"(r1[3]));

asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(r2[0]), "=f"(r2[1]), "=f"(r2[2]), "=f"(r2[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r2[0]), "f"(r2[1]), "f"(r2[2]), "f"(r2[3]));
```
sass
```
        /*0230*/                   HMMA.16816.F32 R4, R12.reuse, R16, R4 ;    /* 0x000000100c04723c */
                                                                              /* 0x060f700000001804 */
        /*0240*/                   HMMA.16816.F32 R8, R12, R2, R8 ;           /* 0x000000020c08723c */
                                                                              /* 0x000f5e0000001808 */
        /*0250*/                   NOP ;                                      /* 0x0000000000007918 */
                                                                              /* 0x000fc20000000000 */
        /*0260*/                   HMMA.16816.F32 R4, R12.reuse, R16, R4 ;    /* 0x000000100c04723c */
                                                                              /* 0x060f700000001804 */
        /*0270*/                   HMMA.16816.F32 R8, R12, R2, R8 ;           /* 0x000000020c08723c */
                                                                              /* 0x000f5e0000001808 */
        /*0280*/                   NOP ;                                      /* 0x0000000000007918 */
```
70 8 cycle
5e 15 cycle
c2 1 cycle

三个
```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(r1[0]), "=f"(r1[1]), "=f"(r1[2]), "=f"(r1[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r1[0]), "f"(r1[1]), "f"(r1[2]), "f"(r1[3]));

asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(r2[0]), "=f"(r2[1]), "=f"(r2[2]), "=f"(r2[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r2[0]), "f"(r2[1]), "f"(r2[2]), "f"(r2[3]));

asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(r3[0]), "=f"(r3[1]), "=f"(r3[2]), "=f"(r3[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(r3[0]), "f"(r3[1]), "f"(r3[2]), "f"(r3[3]));
```
sass
```
        /*0260*/                   HMMA.16816.F32 R4, R16.reuse, R2.reuse, R4 ;    /* 0x000000021004723c */
                                                                                   /* 0x0c0f700000001804 */
        /*0270*/                   HMMA.16816.F32 R8, R16.reuse, R2.reuse, R8 ;    /* 0x000000021008723c */
                                                                                   /* 0x0e0f700000001808 */
        /*0280*/                   HMMA.16816.F32 R12, R16.reuse, R2.reuse, R12 ;  /* 0x00000002100c723c */
                                                                                   /* 0x0c0f70000000180c */
        /*0290*/                   HMMA.16816.F32 R4, R16.reuse, R2.reuse, R4 ;    /* 0x000000021004723c */
                                                                                   /* 0x0c0f700000001804 */
        /*02a0*/                   HMMA.16816.F32 R8, R16.reuse, R2.reuse, R8 ;    /* 0x000000021008723c */
                                                                                   /* 0x0e0f700000001808 */
        /*02b0*/                   HMMA.16816.F32 R12, R16.reuse, R2.reuse, R12 ;  /* 0x00000002100c723c */
```
f7 8 cycle
f7 8 cycle
f7 8 cycle

Q:
1. 为啥8个周期就可以发射一条mma，如果前后存在数据依赖，则需要隔24个周期

## reference
[相似的问题](https://zhuanlan.zhihu.com/p/720562971)