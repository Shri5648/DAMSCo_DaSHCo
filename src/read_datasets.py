import pickle
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import os
import requests
from torch.utils.data import TensorDataset

def read_datasets(dataset_name, data_dir=None):
    if dataset_name in ["CIFAR10", "FashionMNIST", "Shakespeare","Fineweb"]:
        pass
    else:
        print('New dataset, readdatasets need adjustment')
        return None, None
        

    if data_dir==None:
        data_dir = os.getcwd() + '/data/' + dataset_name + '/'
        os.makedirs(data_dir, exist_ok=True)
        
    if dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
                   
        test_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        return  train_dataset, test_dataset
 
    if dataset_name == "CIFAR10":
    
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        
        return train_dataset, test_dataset

    if dataset_name == "Shakespeare":
        # download the tiny shakespeare dataset
        # print(data_dir)
        input_file_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            # print("input download started")
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            response = requests.get(data_url)
            with open(input_file_path, 'w') as f:
                f.write(response.text)
            # print("input downloaded")

        with open(input_file_path, 'r') as f:
            data = f.read()

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }

        def encode(s):
            return [stoi[c] for c in s] # encoder: take a string, output a list of integers
        def decode(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # create the train and test splits
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode both to integers
        train_ids = encode(train_data)
        val_ids = encode(val_data)

        # convert to tensors
        block_size = 128
        train_data = [train_ids[i:i+block_size] for i in range(0, len(train_ids) - block_size, block_size)]
        train_targets = [train_ids[i+1:i+1+block_size] for i in range(0, len(train_ids) - block_size, block_size)]
        test_data = [val_ids[i:i+block_size] for i in range(0, len(val_ids) - block_size, block_size)]
        test_targets = [val_ids[i+1:i+1+block_size] for i in range(0, len(val_ids) - block_size, block_size)]

        train_data = torch.tensor(train_data, dtype=torch.long)
        train_targets = torch.tensor(train_targets, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.long)
        test_targets = torch.tensor(test_targets, dtype=torch.long)

        # wrap in TensorDataset
        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)


        # Saving meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        return train_dataset, test_dataset

    if dataset_name == "Fineweb":
        """
        Load Fineweb-Edu dataset from pre-tokenized shards.
        Assumes Fineweb.py has already been run to create the shards.
        """
        # Path to the tokenized shards directory
        shard_dir = os.path.join(data_dir, 'edu_Fineweb10B')
        
        # Check if data has been prepared
        if not os.path.exists(shard_dir):
            raise FileNotFoundError(
                f"Fineweb shards not found at {shard_dir}\n"
                f"Please run: python data/Fineweb/Fineweb.py first to download and tokenize the data."
            )
        
        # Find all shard files
        train_shards = sorted(glob.glob(os.path.join(shard_dir, 'eduFineweb_train_*.npy')))
        val_shards = sorted(glob.glob(os.path.join(shard_dir, 'eduFineweb_val_*.npy')))
        
        if len(train_shards) == 0 or len(val_shards) == 0:
            raise FileNotFoundError(
                f"No shard files found in {shard_dir}\n"
                f"Expected files like: eduFineweb_train_000001.npy, eduFineweb_val_000000.npy"
            )
        
        print(f"Found {len(train_shards)} training shards and {len(val_shards)} validation shards")
        
        # Load and concatenate all shards for training and validation
        # Note: For very large datasets, you might want to load shards on-demand instead
        train_data_list = []
        for shard_path in train_shards:
            shard_data = np.load(shard_path)
            train_data_list.append(shard_data)
        train_data_np = np.concatenate(train_data_list)
        
        val_data_list = []
        for shard_path in val_shards:
            shard_data = np.load(shard_path)
            val_data_list.append(shard_data)
        val_data_np = np.concatenate(val_data_list)
        
        print(f"Loaded {len(train_data_np):,} training tokens and {len(val_data_np):,} validation tokens")
        
        # Create sequences with block_size context length
        block_size = 1024  # Standard for Fineweb (BPE tokens, not characters)
        
        # Training sequences
        train_sequences = []
        train_targets = []
        for i in range(0, len(train_data_np) - block_size, block_size):
            train_sequences.append(train_data_np[i:i+block_size])
            train_targets.append(train_data_np[i+1:i+1+block_size])
        
        # Validation sequences
        val_sequences = []
        val_targets = []
        for i in range(0, len(val_data_np) - block_size, block_size):
            val_sequences.append(val_data_np[i:i+block_size])
            val_targets.append(val_data_np[i+1:i+1+block_size])
        
        # Convert to tensors
        train_data = torch.tensor(np.array(train_sequences), dtype=torch.long)
        train_targets = torch.tensor(np.array(train_targets), dtype=torch.long)
        test_data = torch.tensor(np.array(val_sequences), dtype=torch.long)
        test_targets = torch.tensor(np.array(val_targets), dtype=torch.long)
        
        print(f"Created {len(train_data):,} training examples and {len(test_data):,} validation examples")
        
        # Wrap in TensorDataset
        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)
        
        # Save meta information (GPT-2 BPE tokenizer)
        meta = {
            'vocab_size': 50304,  # GPT-2 vocab size (50257) rounded up for efficiency
            'tokenizer': 'gpt2_bpe',  # Using tiktoken GPT-2 BPE
            'block_size': block_size,
        }
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        
        return train_dataset, test_dataset 
     
