"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

3) https://github.com/karpathy/nanoGPT/blob/master/
"""

from .gpt import GPT, GPTConfig
import os
import pickle
import torch

__all__=['nanoGPT']

#exec(open('./config/train_shakespeare_char.py').read()) # overrides from your config file. Just for Shakespeare dataset
exec(open('./config/train_fineweb_config.py').read()) # Just for fineweb dataset.

# attempt to derive vocab_size from the dataset
#data_dir = './data/Shakespeare' # to be changed everytime if used a different setting.  Just for Shakespeare dataset.
data_dir = './data/edu_fineweb10B'
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# Check if init_from is defined in the config file
init_from = globals().get('init_from', 'scratch')

if init_from == 'scratch':
    # print("Initializing a new model from scratch")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=dropout
    )
    gptconf = GPTConfig(**model_args)
elif init_from == 'resume':
    ckpt_path = "" # add the checkpoint file path here
    checkpoint = torch.load(ckpt_path)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired
    model_args = {}
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size',  'vocab_size']:#'bias',
        model_args[k] = checkpoint_model_args[k]
    # create the model
    print("Model vocab size:", model_args['vocab_size'])
    gptconf = GPTConfig(**model_args)
else:
    raise ValueError(f"Unknown init_from: {init_from}. Expected 'scratch' or 'gpt2-*'")

new_out_dir = "./results/"+globals().get('out_dir')

class nanoGPT(GPT):
    def __init__(self):
        super().__init__(gptconf)
        self.config = gptconf
        self.out_dir = new_out_dir
        if block_size < self.config.block_size:
            self.crop_block_size(block_size)
            model_args['block_size'] = block_size
        self.model_args = model_args
        
        if init_from == 'resume':
            state_dict = checkpoint['model']
            super().load_state_dict(state_dict)
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']
