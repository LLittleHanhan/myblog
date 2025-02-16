---
title: lab4
date: 2024-08-24
tags: os.mit6.828
---
lab4完成
1. 开启多处理器
2. 多核（多cpu）进程调度
3. 进程创建
4. 进程通信
<!--more-->

## boot mp
```c
void i386_init(void) {
    extern char edata[], end[];
    // Before doing anything else, complete the ELF loading process.
    // Clear the uninitialized global data (BSS) section of our program.
    // This ensures that all static/global variables start out zero.
    memset(edata, 0, end - edata);
    // Initialize the console.
    // Can't call cprintf until after we do this!
    cons_init();
    // Lab 2 memory management initialization functions
    mem_init();
    // Lab 3 user environment initialization functions 
    env_init();
    trap_init();
    // Lab 4 multiprocessor initialization functions 
    mp_init(); 
    lapic_init();
    // Lab 4 multitasking initialization functions
    pic_init();

    // Acquire the big kernel lock before waking up APs
    // Your code here:
    lock_kernel();
    boot_aps();
```
这块牵扯到硬件的内容，不过多描述
1. bios在启动时会检查硬件制作mp configuration table，这张表中保存了lapic的物理地址
2. 哪一个CPU是BSP由硬件和BISO决定，之前实验的JOS代码都运行在BSP上。在SMP系统中，每个CPU都有一个对应的local APIC（LAPIC），负责传递中断。CPU通过内存映射IO(MMIO)访问它对应的APIC，这样就能通过访问内存达到访问设备寄存器的目的。
3. bios制作了floating pointer（实际是一个结构体），保存mp configuration table的物理地址，而floating pointer被规定保存在某个固定位置
4. mp_init函数：通过floating pointer找到mp configuration table，通过mp configuration table的信息找到bsp和ap，并填充struct CpuInfo cpus[NCPU]的id
   ```c
    // Per-CPU state
    struct CpuInfo {
        uint8_t cpu_id;                 // Local APIC ID; index into cpus[] below
        volatile unsigned cpu_status;   // The status of the CPU
        struct Env *cpu_env;            // The currently-running environment.
        struct Taskstate cpu_ts;        // Used by x86 to find stack for interrupt
    };
   ```
5. lapic_init函数：`lapic = mmio_map_region(lapicaddr, 4096);` 给lapic的内存区域做映射

### boot_aps
这个函数启动了每个ap，并做初始化工作，和bsp一样，启动时处于实模式，需要将启动代码拷贝到1MB一下的位置具体是0x7000，步骤和bsp基本一致
具体代码在mpentry.S中mpentry.S之后调用mp_main
> 注意内核栈，unsigned char percpu_kstacks[NCPU][KSTKSIZE]声明了所有cpu的内核，但是bsp初始化时使用的是bootstack
```c
// Setup code for APs
void mp_main(void) {
    // We are in high EIP now, safe to switch to kern_pgdir
    lcr3(PADDR(kern_pgdir));
    cprintf("SMP: CPU %d starting\n", cpunum());

    lapic_init();
    env_init_percpu();// 装载gdtr，lgdtr
    trap_init_percpu();//装载tr
    xchg(&thiscpu->cpu_status, CPU_STARTED);  // tell boot_aps() we're up
}
```
1. 所有cpu共用一个gdt，pages，envs
2. 每个cpu有各自的tss段描述符gdt，和tss段(struct Cpuinfo->struct Taskstate cpu_ts)
3. 在里实际上就是多线程了，bsp启动boot_aps，在boot_aps中启动每个cpu执行mpentry.S+mp_main，每个ap在mp_main把cpu_status设置为started，bsp轮询检查，检查通过后启动下一个ap
4. 一个cpu启动主要需要做以下工作
   1. gdt表开启分段
   2. cr3页表开启分页
   3. idt表实现中断
   4. tss表实现中断时内核栈转换
> 在ap启动时会发现它使用的页表是最开始的entry_pgdir，之后不变直到运行用户进程时加载用户页目录表，创建进程时用户的页目录实际是从内核页目录复制过来的


## 进程调度
实验中使用的是轮询

## 创建进程，写时复制

### pagefault
_pgfault_upcall -> _pgfault_handler
set_pgfault_handler : sys_env_set_pgfault_upcall(0, _pgfault_upcall):env->env_pgfault_upcall = func;

## 进程通信
