# vllm之执行
```python
execute_model_req = ExecuteModelRequest(
    seq_group_metadata_list=seq_group_metadata_list,
    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
    blocks_to_copy=scheduler_outputs.blocks_to_copy,
    num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
    running_queue_size=scheduler_outputs.running_queue_size,
    finished_requests_ids=finished_requests_ids,
    last_sampled_token_ids=last_sampled_token_ids)

outputs = self.model_executor.execute_model(
    execute_model_req=execute_model_req)
```