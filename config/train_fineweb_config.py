out_dir = 'out-fineweb'
block_size: int = 1024 # max sequence length
vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
n_layer: int = 12 # number of layers
n_head: int = 12 # number of heads
n_embd: int = 768 # embedding dimension
dropout=0.1

batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
ddp_world_size = int(os.environ['WORLD_SIZE'])
assert batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
gradient_accumulation_steps = batch_size // (B * T * ddp_world_size)
print(f"total desired batch size: {batch_size}")
print(f"=> calculated gradient accumulation steps: {gradient_accumulation_steps}")
