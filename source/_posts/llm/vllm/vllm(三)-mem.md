# vllm之内存

## 源码走读
输入的str首先转成Sequence类型，接着转成SequenceGroup类型
_schedule_prefills阶段，经过一系列检查就可以调用_allocate_and_set_running函数给sg分配block
可以看到该函数做了两件事
1. 调用LLMEngine.Scheduler.block_manager对象分配块
2. 把sg中的每一个seq状态由waiting改为running
### data structure
核心的数据结构层次如下
![alt text](image-1.png)
- Scheduler
    - SelfAttnBlockSpaceManager
      - CpuGpuBlockAllocator
      - Dict[SeqId, BlockTable]
        - Block

SelfAttnBlockSpaceManager中包含一个CpuGpuBlockAllocator和一个Dict[SeqId, BlockTable]
SelfAttnBlockSpaceManager的作用是manages the allocation of KV cache，能力由CpuGpuBlockAllocator和BlockTable提供
```python
class Scheduler:
    self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

class SelfAttnBlockSpaceManager(BlockSpaceManager):
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        assert watermark >= 0.0
        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )
        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}
```

一个seq对应一个BlockTable，分配的过程就是建立或修改这张表
核心的数据结构self._blocks: BlockList记录了当前seq分配的逻辑block
_num_full_slots记录了table的token数即slot
```python
class BlockTable:
    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,# 就是上层SelfAttnBlockSpaceManager的block_allocator
        _blocks: Optional[List[Block]] = None,
        max_block_sliding_window: Optional[int] = None,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        if _blocks is None:
            _blocks = []
        self._blocks: BlockList = BlockList(_blocks)

        self._max_block_sliding_window = max_block_sliding_window
        self._num_full_slots = self._get_num_token_ids()

class BlockList:
    def __init__(self, blocks: List[Block]):
        self._blocks: List[Block] = []
        self._block_ids: List[int] = []
        self.update(blocks)
```

CpuGpuBlockAllocator，统合了设备和主机的分配器，核心数据结构
- gpu_allocator & cpu_allocator
- _swap_mapping
```python
class CpuGpuBlockAllocator(DeviceAwareBlockAllocator):
    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:

        block_ids = list(range(num_gpu_blocks + num_cpu_blocks))
        # 可以看到他把gpu和cpu的block统一编id
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        if allocator_type == "naive":
            gpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock, 
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        elif allocator_type == "prefix_caching":
            gpu_allocator = PrefixCachingBlockAllocator(
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator = PrefixCachingBlockAllocator(
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return CpuGpuBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(self, cpu_block_allocator: BlockAllocator,
                gpu_block_allocator: BlockAllocator):
    assert not (
        cpu_block_allocator.all_block_ids
        & gpu_block_allocator.all_block_ids
    ), "cpu and gpu block allocators can't have intersection of block ids"

    self._allocators = {
        Device.CPU: cpu_block_allocator,
        Device.GPU: gpu_block_allocator,
    }

    self._swap_mapping: Dict[int, int] = {}
    self._null_block: Optional[Block] = None

    self._block_ids_to_allocator: Dict[int, BlockAllocator] = {}
    for _, allocator in self._allocators.items():
        for block_id in allocator.all_block_ids:
            self._block_ids_to_allocator[block_id] = allocator

```
NaiveBlockAllocator的核心数据结构
- _free_block_indices：空闲block id，双端队列
- _all_block_indices：所有block id，集合
- _refcounter：给_free_block_indices建立引用计数
- block_pool：提前创建block，避免运行时创建的开销
```python
class NaiveBlockAllocator(BlockAllocator):
    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
        block_pool: Optional[BlockPool] = None,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)

        self._free_block_indices: Deque[BlockId] = deque(block_ids)
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks

        self._refcounter = RefCounter(
            all_block_indices=self._free_block_indices)
        self._block_size = block_size

        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly())

        if block_pool is None:
            extra_factor = 4
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks
            self._block_pool = BlockPool(self._block_size, create_block, self,
                                         num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool

class BlockPool:
    def __init__(self, block_size: int, create_block: Block.Factory,
                 allocator: BlockAllocator, pool_size: int):
        self._block_size = block_size
        self._create_block = create_block
        self._allocator = allocator
        self._pool_size = pool_size
        assert self._pool_size >= 0
        self._free_ids: Deque[int] = deque(range(self._pool_size))
        self._pool = []
        for i in range(self._pool_size):
            self._pool.append(
                self._create_block(prev_block=None,
                                   token_ids=[],
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None))
```

block: 包含了一个block的信息，核心关注一个是_prev_block，一个是_cow_target
写时复制的策略之后再说copy on write(cow)
```python
class NaiveBlock(Block):
    def __init__(self,
                 prev_block: Optional[Block],
                 token_ids: List[int],
                 block_size: int,
                 allocator: BlockAllocator,
                 block_id: Optional[int] = None,
                 _cow_target: Optional[Block] = None):
        self._token_ids: List[int] = []
        self._block_size = block_size
        self._prev_block = prev_block
        self._block_id = block_id
        self._allocator = allocator
        self._cow_target = _cow_target if _cow_target is not None else self

        self._append_token_ids_no_cow(token_ids)
```
### summary
这里主要有两条线
1. block线，关注如何管理所有的block
NaiveBlock存储了block的元信息，GPU和CPU分别对应一个NaiveBlockAllocator，用于管理NaiveBlock
CpuGpuBlockAllocator包含了GPU和CPU的NaiveBlockAllocator，应该是用于处理swap
2. seq线，关注如何做表
每个seq对应一个List[Block]

具体怎么映射物理内存上呢?

### control

#### 初始化时
1. 调用_allocate_sequence给seq[0]制作block table
2. 复制给其他seq，因为一个seq group中的seq共享一个prompt
```python
# SelfAttnBlockSpaceManager.allocate
def allocate(self, seq_group: SequenceGroup) -> None:
    waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
    # NOTE: Here we assume that all sequences in the group have the same
    # prompt.
    seq = waiting_seqs[0]
    block_table: BlockTable = self._allocate_sequence(seq)
    self.block_tables[seq.seq_id] = block_table
    # Assign the block table for each sequence.
    for seq in waiting_seqs[1:]:
        self.block_tables[seq.seq_id] = block_table.fork()

# SelfAttnBlockSpaceManager._allocate_sequence
def _allocate_sequence(self, seq: Sequence) -> BlockTable:
    block_table = BlockTable(
        block_size=self.block_size,
        block_allocator=self.block_allocator,
        max_block_sliding_window=self.max_block_sliding_window,
    )
    if seq.get_token_ids():
        # Add blocks to the block table only if the sequence is non empty.
        block_table.allocate(seq.get_token_ids())

    return block_table
```
```python
# BlockTable.allocate
def allocate(self,
                 token_ids: List[int],
                 device: Device = Device.GPU) -> None:
    blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                    token_ids=token_ids,
                                                    device=device)
    self.update(blocks)
    self._num_full_slots = len(token_ids)
    
# BlockTable._allocate_blocks_for_token_ids
def _allocate_blocks_for_token_ids(self, prev_block: Optional[Block],
                                       token_ids: List[int],
                                       device: Device) -> List[Block]:
        blocks: List[Block] = []

        block_token_ids = []
        tail_token_ids = []
        for cur_token_ids in chunk_list(token_ids, self._block_size):
            if len(cur_token_ids) == self._block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_token_ids.append(cur_token_ids)

        if block_token_ids:
            blocks.extend(
                self._allocator.allocate_immutable_blocks(
                    prev_block, block_token_ids=block_token_ids,
                    device=device))
            prev_block = blocks[-1]

        if tail_token_ids:
            assert len(tail_token_ids) == 1
            cur_token_ids = tail_token_ids[0]

            block = self._allocator.allocate_mutable_block(
                prev_block=prev_block, device=device)
            block.append_token_ids(cur_token_ids)

            blocks.append(block)

        return blocks
```
```python
# CpuGpuBlockAllocator.allocate_immutable_blocks
# CpuGpuBlockAllocator.allocate_immutable_block
# CpuGpuBlockAllocator.allocate_mutable_block
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NaiveBlockAllocator.allocate_immutable_blocks
def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: Optional[Device] = None) -> List[Block]:
    assert device is None
    num_blocks = len(block_token_ids)

    block_ids = []
    for i in range(num_blocks):
        block_ids.append(self._allocate_block_id())

    blocks = []
    for i in range(num_blocks):
        prev_block = self._block_pool.init_block(
            prev_block=prev_block,
            token_ids=block_token_ids[i],
            block_size=self._block_size,
            physical_block_id=block_ids[i])
        blocks.append(prev_block)
    return blocks

def _allocate_block_id(self) -> BlockId:
    if not self._free_block_indices:
        raise BlockAllocator.NoFreeBlocksError()

    block_id = self._free_block_indices.popleft()
    self._refcounter.incr(block_id)
    return block_id
# NaiveBlockAllocator.allocate_immutable_block
# NaiveBlockAllocator.allocate_mutable_blocks
```
##### summary
1. 从空闲队列中取block id
2. 向block pool中申请block
3. 更改block id的引用计数

不同的block可能具有相同的block id即共享同一片内存，这里可以解释为什么block pool要过量申请block，即block是逻辑block，block id表示物理block


#### 抢占时
两种抢占策略
##### _preempt_by_recompute
1. free seq
2. 修改seq状态
```python
# Scheduler
def _preempt_by_recompute(
    self,
    seq_group: SequenceGroup,
) -> None:
    seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    assert len(seqs) == 1
    for seq in seqs:
        seq.status = SequenceStatus.WAITING
        self.free_seq(seq)
        seq.reset_state_for_recompute()
```
1. table调用alloctor管理block
2. 删除table
```python
# SelfAttnBlockSpaceManager
def free(self, seq: Sequence) -> None:
    seq_id = seq.seq_id
    if seq_id not in self.block_tables:
        return
    self.block_tables[seq_id].free()
    del self.block_tables[seq_id]
```
table调用alloctor
```python
# BlockTable
def free(self) -> None:
    for block in self._blocks:
        self._allocator.free(block)
    self._blocks.reset()
```
alloctor调用负责该block的子alloctor
```python
# CpuGpuBlockAllocator
def free(self, block: Block) -> None:
    block_id = block.block_id
    assert block_id is not None
    allocator = self._block_ids_to_allocator[block_id]
    allocator.free(block)
```
1. 减少引用计数
2. 如果为0,把block重新添加到空闲链表
```python
# NaiveBlockAllocator
def free(self, block: Block, keep_block_object: bool = False) -> None:
    self._free_block_id(block)
    if not keep_block_object:
        self._block_pool.free_block(block)

def _free_block_id(self, block: Block) -> None:
    block_id = block.block_id
    assert block_id is not None
    refcount = self._refcounter.decr(block_id)
    if refcount == 0:
        self._free_block_indices.appendleft(block_id)
    block.block_id = None
```
##### _preempt_by_swap
1. swap out
2. 修改seq状态
```python
# Scheduler
def _preempt_by_swap(
    self,
    seq_group: SequenceGroup,
    blocks_to_swap_out: List[Tuple[int, int]],
) -> None:
    self._swap_out(seq_group, blocks_to_swap_out)

def _swap_out(
    self,
    seq_group: SequenceGroup,
    blocks_to_swap_out: List[Tuple[int, int]],
) -> None:
    if not self.block_manager.can_swap_out(seq_group):
        raise RuntimeError(
            "Aborted due to the lack of CPU swap space. Please increase "
            "the swap space to avoid this error.")
    mapping = self.block_manager.swap_out(seq_group)
    blocks_to_swap_out.extend(mapping)
    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
        seq.status = SequenceStatus.SWAPPED
```
```python
# SelfAttnBlockSpaceManager
def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
    physical_block_id_mapping = []
    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
        blocks = self.block_tables[seq.seq_id].blocks
        if len(blocks) == 0:
            continue
        seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                        src_device=Device.GPU,
                                                        dst_device=Device.CPU)

        # Refresh the block ids of the table (post-swap)
        self.block_tables[seq.seq_id].update(blocks)
        seq_physical_block_id_mapping = {
            self.block_allocator.get_physical_block_id(
                Device.GPU, gpu_block_id):
            self.block_allocator.get_physical_block_id(
                Device.CPU, cpu_block_id)
            for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
        }
        physical_block_id_mapping.extend(
            list(seq_physical_block_id_mapping.items()))
    return physical_block_id_mapping
```

```python
# BlockTable
```

```python
# CpuGpuBlockAllocator
def swap(self, blocks: List[Block], src_device: Device,
            dst_device: Device) -> Dict[int, int]:
    src_block_ids = [block.block_id for block in blocks]
    self._allocators[src_device].swap_out(blocks)
    self._allocators[dst_device].swap_in(blocks)
    dst_block_ids = [block.block_id for block in blocks]

    current_swap_mapping: Dict[int, int] = {}
    for src_block_id, dst_block_id in zip(src_block_ids, dst_block_ids):
        if src_block_id is not None and dst_block_id is not None:
            self._swap_mapping[src_block_id] = dst_block_id
            current_swap_mapping[src_block_id] = dst_block_id
    return current_swap_mapping
```
```python
# NaiveBlockAllocator
def swap_out(self, blocks: List[Block]) -> None:
    for block in blocks:
        self._free_block_id(block)

def swap_in(self, blocks: List[Block]) -> None:
    for block in blocks:
        if block.is_full:
            tmp_block = self.allocate_immutable_block(
                prev_block=block.prev_block, token_ids=block.token_ids)
        else:
            tmp_block = self.allocate_mutable_block(
                prev_block=block.prev_block)
            tmp_block.append_token_ids(block.token_ids)

        block_id = tmp_block.block_id
        tmp_block.block_id = None
        self._block_pool.free_block(tmp_block)
        block.block_id = block_id  # Assign block_id
```
summary：
这里有个细节问题，block id是统一分配，实际要减去设备初始的偏移值