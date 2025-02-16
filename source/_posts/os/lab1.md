---
title: lab1 os启动
date: 2024-07-20
tags: os.mit6.828
---
> 大名鼎鼎的mit6.828
> 实验基于i386,32bit系统，一些东西和现在的x86-64bit(amd64)不一样

## x86地址空间划分
```shell

+------------------+  <- 0xFFFFFFFF (4GB)
|      32-bit      |
|  memory mapped   |
|     devices      |
|                  |
/\/\/\/\/\/\/\/\/\/\

/\/\/\/\/\/\/\/\/\/\
|                  |
|      Unused      |
|                  |
+------------------+  <- depends on amount of RAM
|                  |
|                  |
| Extended Memory  |
|                  |
|                  |
+------------------+  <- 0x00100000 (1MB)
|     BIOS ROM     |
+------------------+  <- 0x000F0000 (960KB)
|  16-bit devices, |
|  expansion ROMs  |
+------------------+  <- 0x000C0000 (768KB)
|   VGA Display    |
+------------------+  <- 0x000A0000 (640KB)
|                  |
|                  |
|                  |
+------------------+  <- 0x7c00 (31kb)
|    Low Memory    |
|                  |
+------------------+  <- 0x00000000
```
- 8086,16bit,有20bit地址线，内存访问模式为实模式，cs:ip，寻址空间为1MB
- 之后的32bit处理器，有32bit地址线，寻址空间4GB，为了兼容之前的架构，前1MB保持值先的划分，剩下的作为内存

## bios
启动时为实模式，cs:ip有默认值，一般在那个地址位置放置一条长跳转（bios中），开始正式执行bios程序

bios做的最主要工作
1. 查询硬件信息
2. 加载bootloader

bootloader程序存储在硬盘的第一个扇区512B


## bootloader
bios会把mbr加载到0x7c00的位置(31kb)，之后的bootloader从这个位置执行
> 古早的一种型号计算机内存只有32kb，为了让操作系统有更多的连续空间加载，因此会把bootloader放到31kb的位置，前512B用于加载bootloader，后512B用于运行内存，31kb的位置就是地址0x7c00

bootloader做两件事
1. 进入保护模式（gdt表中设置的段基址为0，所以进了跟没进一样）
2. 加载操作系统到0x100000的位置，即1MB以上的地址空间

### 源码分析/boot/
1. makefrag：这里指定了加载位置
```
$(OBJDIR)/boot/boot: $(BOOT_OBJS)
	@echo + ld boot/boot
	$(V)$(LD) $(LDFLAGS) -N -e start -Ttext 0x7C00 -o $@.out $^
	$(V)$(OBJDUMP) -S $@.out >$@.asm
	$(V)$(OBJCOPY) -S -O binary -j .text $@.out $@
	$(V)perl boot/sign.pl $(OBJDIR)/boot/boot
```
2. boot.S
```asm
# Set up the stack pointer and call into C.
movl    $start, %esp
call bootmain
# 调用bootmain前把esp调到start即代码的开始位置0x7c00


gdt:
  SEG_NULL				# null seg
  SEG(STA_X|STA_R, 0x0, 0xffffffff)	# code seg
  SEG(STA_W, 0x0, 0xffffffff)	        # data seg

gdtdesc:
  .word   0x17                            # sizeof(gdt) - 1
  .long   gdt                             # address gdt
# 这里使用宏做了一个gdt表开启分页模式，只有三个表项，段基址设为0
```
3. main.c中主要是读取内核elf文件的header获取信息，之后加载kernel到固定位置
```c
((void (*)(void)) (ELFHDR->e_entry))();
// bootmain最后调用ELFHDR->e_entry函数，即kernel的入口地址
```


## kernel
见lab2


## print输出
略

## reference
[mit](https://pdos.csail.mit.edu/6.828/2018/labs/lab1/)
[reference](https://www.cnblogs.com/gatsby123/p/9759153.html)