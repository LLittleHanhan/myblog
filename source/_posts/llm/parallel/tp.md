# 张量并行

## 策略
![alt text](../pic/tp-1.png)
张量并行有两种策略X*A=Y
## A矩阵按行切，这时X矩阵需要按列切，最后做all-reduce操作
![alt text](image-3.png)
![alt text](image-4.png)
## A矩阵按列切，最后做all-gather操作
![alt text](image-5.png)
![alt text](image-6.png)

## transformer
![alt text](../pic/tp-2.png)
### MLP层 
X * A + gelu -> Y 
Y * B + droupout -> O
对A按列切得到[Y1,Y2]之后符合对B按行切的形态
![alt text](image-7.png)
### selfattention
多头天然符合张量并行策略
![alt text](image-8.png)

## summary
![alt text](image-9.png)
对bs * seq * dim的数据量做4次all reduce 