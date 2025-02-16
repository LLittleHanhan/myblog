# vllm之调度

## 源码走读
> 这里仅梳理官方example的执行路径

具体的函数调用如下
- LLMEngine.step()
- LLMEngine.scheduler.schedule()
- _schedule()
- _schedule_default()
  - _schedule_frefills()
### data structure
```python
class SchedulingBudget:
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

class ScheduledSequenceGroup:
    seq_group: SequenceGroup
    token_chunk_size: int

class SchedulerPrefillOutputs:
    # Selected sequences for prefill.
    seq_groups: List[ScheduledSequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

class SchedulerRunningOutputs:
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Optimization for fast-access to seq_group lists
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

class SchedulerSwappedInOutputs:
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]
```

### _schedule_default()
```python
budget = SchedulingBudget(
    token_budget=self.scheduler_config.max_num_batched_tokens,
    max_num_seqs=self.scheduler_config.max_num_seqs,
)
for seq_group in self.running:
    budget.add_num_seqs(seq_group.request_id,
                        seq_group.get_max_num_running_seqs())

# 一
if not self.swapped:
    prefills = self._schedule_prefills(budget,curr_loras,enable_chunking=False)
# 二
if len(prefills.seq_groups
        ) == 0 and self.scheduler_config.policy == "priority":
    self._schedule_priority_preemption(budget)
# 三
if len(prefills.seq_groups) == 0:
    running_scheduled = self._schedule_running(budget,
                                                curr_loras,
                                                enable_chunking=False)

    # If any sequence group is preempted, do not swap in any sequence
    # group. because it means there's no slot for new running requests.
    if len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out) == 0:
# 四
        swapped_in = self._schedule_swapped(budget, curr_loras)
```

#### 一. _schedule_prefills()
```python
waiting_queue = self.waiting
while self._passed_delay(time.time()) and waiting_queue:
    seq_group = waiting_queue[0]
    waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
    # seq_group的token数
    num_new_tokens = self._get_num_new_tokens(seq_group,
                                                SequenceStatus.WAITING,
                                                enable_chunking, budget)
    can_allocate = self.block_manager.can_allocate(
                                                seq_group, num_lookahead_slots=num_lookahead_slots)

    num_new_seqs = seq_group.get_max_num_running_seqs()
    if (num_new_tokens == 0
            or not budget.can_schedule(num_new_tokens=num_new_tokens,
            num_new_seqs=num_new_seqs)):
        break
    
    # 可调度和分配
    waiting_queue.popleft()
    self._allocate_and_set_running(seq_group)
    # 这里还有一部分似乎和multistep有关，先不管
    seq_groups.append(
            ScheduledSequenceGroup(seq_group=seq_group,
                                    token_chunk_size=num_new_tokens))
        budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
        budget.add_num_seqs(seq_group.request_id, num_new_seqs)
    
return SchedulerPrefillOutputs(
        seq_groups=seq_groups,
        ignored_seq_groups=ignored_seq_groups,
        num_lookahead_slots=self._get_num_lookahead_slots( # 这玩意儿是啥o:O?
            is_prefill=True, enable_chunking=enable_chunking))
```
##### 1. _passed_delay
```python
```
##### 2. SelfAttnBlockSpaceManager.can_allocate
该函数返回三个状态
- OK：可以分配
- LATER:稍后分配
- NEVER：无法分配
具体策略
total-required < watermark -------- Never
free -required < watermark -------- Later
```python
seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
num_required_blocks = BlockTable.get_num_required_blocks(
    seq.get_token_ids(),
    block_size=self.block_size,
    num_lookahead_slots=num_lookahead_slots,
)
num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)
# Use watermark to avoid frequent cache eviction.
if (self.num_total_gpu_blocks - num_required_blocks <
        self.watermark_blocks):
    return AllocStatus.NEVER
if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
    return AllocStatus.OK
else:
    return AllocStatus.LATER
```
##### 3. SchedulingBudget.can_schedule
```python
def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
    assert num_new_tokens != 0
    assert num_new_seqs != 0
    return (self.num_batched_tokens + num_new_tokens <= self.token_budget
            and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)
```
##### 4. Scheduler._allocate_and_set_running
```python
def allocate(self, seq_group: SequenceGroup) -> None:
    self.block_manager.allocate(seq_group)
    for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        seq.status = SequenceStatus.RUNNING
``` 

#### 二. _schedule_priority_preemption
这部分代码逻辑十分清晰，
取waiting的第一个sq（保证是最高的优先级），从running的最低优先级的sq比较，若waiting的第一个sq的优先级大于rnning的sq，则从budget中取消掉running的这个sq，如果取消后的资源可以分配waiting的第一个sq，则结束，否则循环

取消的sq调用_preempt函数，具体的抢占策略见后
```python
def _schedule_priority_preemption(
    self,
    budget: SchedulingBudget,
) -> int:
    waiting_queue = self.waiting
    running_queue = deque(sorted(self.running, key=self._get_priority))

    blocks_to_swap_out: List[Tuple[int, int]] = []
    force_preemption_count = 0

    if waiting_queue:
        seq_group = waiting_queue.popleft()
        num_new_seqs = seq_group.get_max_num_running_seqs()
        num_new_tokens = self._get_num_new_tokens(seq_group,
                                                    SequenceStatus.WAITING,
                                                    False, budget)

        #Only preempt if priority inversion exists
        while running_queue and self._get_priority(
                running_queue[-1]) > self._get_priority(seq_group):
            #Only preempt if waiting sequence cannot be allocated
            can_allocate = self.block_manager.can_allocate(seq_group)
            if (num_new_tokens and can_allocate == AllocStatus.OK
                    and budget.can_schedule(num_new_tokens=num_new_tokens,
                                            num_new_seqs=num_new_seqs)):
                break

            #Adjust budget to remove the victim sequence group
            vseq_group = running_queue.pop()
            num_running_tokens = self._get_num_new_tokens(
                vseq_group, SequenceStatus.RUNNING, False, budget)
            budget.subtract_num_batched_tokens(vseq_group.request_id,
                                                num_running_tokens)
            num_running_seqs = vseq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(vseq_group.request_id,
                                        num_running_seqs)

            #Preempt out the victim sequence group
            self._preempt(vseq_group, blocks_to_swap_out,
                            PreemptionMode.RECOMPUTE)
            waiting_queue.appendleft(vseq_group)
            force_preemption_count += 1
        #Put the sequence back into the waiting queue
        waiting_queue.appendleft(seq_group)

    waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))

    self.waiting = waiting_queue
    self.running = running_queue
    return force_preemption_count
```


#### 三. _schedule_running
running的调度逻辑
循环取running队列的sq，如果不可扩展（kv cache），则依次preempt running队列的尾端sq，具体策略是recompute或swap out

```python
def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
) -> SchedulerRunningOutputs:
    
    while running_queue:
        seq_group = running_queue[0]
        num_running_tokens = self._get_num_new_tokens(
            seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

        if num_running_tokens == 0:
            # No budget => Stop
            break

        running_queue.popleft()

        while not self._can_append_slots(seq_group, enable_chunking):
            budget.subtract_num_batched_tokens(seq_group.request_id,
                                                num_running_tokens)
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                        num_running_seqs)
            # Determine victim sequence
            cont_loop = True
            if running_queue:
                victim_seq_group = running_queue.pop()
            else:
                victim_seq_group = seq_group
                cont_loop = False

            # Do preemption
            do_preempt = True
            if do_preempt:
                preempted_mode = self._preempt(victim_seq_group,
                                                blocks_to_swap_out)
                if preempted_mode == PreemptionMode.RECOMPUTE:
                    preempted.append(victim_seq_group)
                else:
                    swapped_out.append(victim_seq_group)

            if not cont_loop:
                break
        else:
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            is_prefill = seq_group.is_prefill()

            if is_prefill:
                ret.prefill_seq_groups_list.append(seq_group)
            else:
                ret.decode_seq_groups_list.append(seq_group)

            budget.add_num_batched_tokens(seq_group.request_id,
                                            num_running_tokens)
    self._scheduler_running_outputs_cache[self.next_cache_id].reset()
    self._scheduled_seq_group_cache[self.next_cache_id].reset()
    return ret
```

##### _preempt
```python
# self._preempt_by_recompute(seq_group)
# self._preempt_by_swap(seq_group, blocks_to_swap_out)
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

def _preempt_by_swap(
    self,
    seq_group: SequenceGroup,
    blocks_to_swap_out: List[Tuple[int, int]],
) -> None:
    self._swap_out(seq_group, blocks_to_swap_out)
```

这里有个问题，budget的状态
runninn_schedule的状态转移很奇怪

#### 四. _schedule_swapped
这部分粗略扫了一眼
```python
def _schedule_swapped(
    self,
    budget: SchedulingBudget,
    curr_loras: Optional[Set[int]],
    enable_chunking: bool = False,
) -> SchedulerSwappedInOutputs:
    blocks_to_swap_in: List[Tuple[int, int]] = []
    blocks_to_copy: List[Tuple[int, int]] = []
    decode_seq_groups: List[ScheduledSequenceGroup] = []
    prefill_seq_groups: List[ScheduledSequenceGroup] = []
    infeasible_seq_groups: List[SequenceGroup] = []
    swapped_queue = self.swapped
    leftover_swapped: Deque[SequenceGroup] = deque()

    while swapped_queue:
        seq_group = swapped_queue[0]
        # If the sequence group cannot be swapped in, stop.
        is_prefill = seq_group.is_prefill()
        alloc_status = self.block_manager.can_swap_in(
            seq_group,
            self._get_num_lookahead_slots(is_prefill, enable_chunking))
        if alloc_status == AllocStatus.LATER:
            break
        elif alloc_status == AllocStatus.NEVER:
            logger.warning(
                "Failing the request %s because there's not enough kv "
                "cache blocks to run the entire sequence.",
                seq_group.request_id)
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.FINISHED_IGNORED
            infeasible_seq_groups.append(seq_group)
            swapped_queue.popleft()
            continue
        # The total number of sequences in the RUNNING state should not
        # exceed the maximum number of sequences.
        num_new_seqs = seq_group.get_max_num_running_seqs()
        num_new_tokens = self._get_num_new_tokens(seq_group,
                                                    SequenceStatus.SWAPPED,
                                                    enable_chunking, budget)
        if (num_new_tokens == 0
                or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                            num_new_seqs=num_new_seqs)):
            break

        swapped_queue.popleft()
        self._swap_in(seq_group, blocks_to_swap_in)
        self._append_slots(seq_group, blocks_to_copy, enable_chunking)
        is_prefill = seq_group.is_prefill()
        if is_prefill:
            prefill_seq_groups.append(
                ScheduledSequenceGroup(seq_group,
                                        token_chunk_size=num_new_tokens))
        else:
            decode_seq_groups.append(
                ScheduledSequenceGroup(seq_group, token_chunk_size=1))
        budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
        budget.add_num_seqs(seq_group.request_id, num_new_seqs)

    swapped_queue.extendleft(leftover_swapped)
    
    return SchedulerSwappedInOutputs(
        decode_seq_groups=decode_seq_groups,
        prefill_seq_groups=prefill_seq_groups,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_copy=blocks_to_copy,
        num_lookahead_slots=self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking),
        infeasible_seq_groups=infeasible_seq_groups,
    )
```

#### summary phase1
第一部分是三个队列的调度
1. 首先判断swap队列是否为空，若为空调度waiting队列，若不为空跳过waiting
2. 若调度waiting的结果为空（调度了但没结果或没调度），调度running队列
3. 调度running时会出现内存不足导致一些sq的status由running->swap
4. 若没出现3则说明空间充足，换入swap

两种排序策略1.fcfs 2.priority

##### 三个队列的变化
1. prefill schedule，可执行的sg从waiting队列左端弹出，返回可执行的SchedulerPrefillOutputs.seq_groups
2. running schedule，可执行的sg从running队列左端弹出，不可执行的sg从右端弹出，返回3个队列
   1. SchedulerRunningOutputs.decode_seq_groups可执行队列
   2. SchedulerRunningOutputs.preempted重计算队列
   3. SchedulerRunningOutputs.swapped_out换出队列
3. swap schedule，可换入队列从swap队列左端弹出，返回两个队列
   1. SchedulerSwappedInOutputs.decode_seq_groups
   2. SchedulerSwappedInOutputs.prefill_seq_groups

之后根据这些输出，对三个队列统一处理

#### _schedule_default的输出
```python
num_prefill_groups = len(prefills.seq_groups)
if num_prefill_groups > 0:
    scheduled_seq_groups = prefills.seq_groups
    scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
else:
    scheduled_seq_groups = running_scheduled.decode_seq_groups
scheduled_seq_groups.extend(swapped_in.decode_seq_groups)
```
1. 当num_prefill_groups>0时，running_scheduled为空，swapped_in（SchedulerSwappedInOutputs）队列为空
2. 当num_prefill_groups=0时，
   1. 没调度，swap队列不为空，
   2. 调度了，swap为空

### schedule
1. schedule调用_schedule_default获取scheduled_seq_groups
2. 制作List[SequenceGroupMetadata]
```python
class SequenceGroupMetadata(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    request_id: str
    is_prompt: bool
    seq_data: Dict[int, SequenceData]
    sampling_params: Optional[SamplingParams]
    block_tables: Dict[int, List[int]]
    do_sample: bool = True
    computed_block_nums: Optional[List[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(
        default_factory=lambda: SequenceGroupState())
    token_chunk_size: Optional[int] = None
```
需要返回的核心数据，seq,block table