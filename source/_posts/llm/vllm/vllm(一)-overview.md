# vllm

## control stream
- llm control stream
  - init()
  - generate()
    - _validate_and_add_requests()
      - _add_request() // loop
        - self.llm_engine.add_request()
          - _add_processed_request()
            - _create_sequence_group_with_sampling
            - add_seq_group
    - _run_engine

## data stream

- init() params: model(str)
  - llm_engine(LLMEngine)
  - request_counter(Counter)
- generate() params: prompts(List[str]),sample_params(SampllingParams)
  - parsed_prompts(Union[PromptType, Sequence[PromptType]] = List[str])
- _validate_and_add_requests() 
- _add_request()
- self.llm_engine.add_request()
  - preprocessed_inputs(DecoderOnlyInputs)
  - processed_inputs(DecoderOnlyInputs)
- _add_processed_request()
  - seq(Sequence)

## core data struction
### Sequence
```python
def __init__(
        self,
        seq_id: int,
        inputs: "SingletonInputs",
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        from_decoder_prompt: bool = True,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = inputs
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        
        self.data = SequenceData.from_seqs(self.prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""
        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None
        # These are used to keep track of delta outputs
        self._last_output_token_ids_offset: int = 0
        self._last_output_text_offset: int = 0
        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None
```

### LLMEngine 
- self.input_preprocessor = InputPreprocessor(model_config,self.tokenizer)
   - self.input_preprocessor.preprocess() -> DecoderOnlyInputs()
     - DecoderOnlyInputs = TokenInputs
        ```python
        class TokenInputs(TypedDict):
            prompt_token_ids: List[int]
            prompt: NotRequired[Optional[str]]
            multi_modal_data: NotRequired[Optional["MultiModalDataDict"]]
            mm_processor_kwargs: NotRequired[Optional[Dict[str, Any]]]
        ```
- self.input_registry = input_registry
- self.input_processor = input_registry.create_input_processor(model_config)
- self.seq_counter = Counter()
- self.cached_scheduler_outputs[SchedulerOutputState() for _ in range(self.parallel_config.pipeline_parallel_size)]
  ```python
  class SchedulerOutputState:
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    allow_async_output_proc: bool = False
    last_output: Optional[SamplerOutput] = None
  ```