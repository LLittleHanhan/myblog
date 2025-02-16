# nsight profile
结合这几篇博客总结profile的方法论
[gemv](https://zhuanlan.zhihu.com/p/715609504?utm_psn=1811339742626856960)

## roofline
优化的第一步，首先需要确定这个问题是计算密集型还是访存密集型
假如是访存密集型算子，其瓶颈在访存，从compute和mem throughout的图中应该可以看到mem throughout的利用率较高，compute单元的利用率较低