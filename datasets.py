import numpy as np
import torch
import glob

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens
import numpy as np
import torch
import glob

# ... (keep _peek_data_shard and _load_data_shard as they were) ...

class BaseDataLoader:
    def __init__(self, filename_pattern, B, T, device):
        self.B = B
        self.T = T
        self.device = device

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"No files found for {filename_pattern}"

                # load and validate all data shards, count total tokens
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # Ensure shard is large enough
            assert shard_ntok >= B * T + 1, f"Shard {fname} too small"
            # FIX: Cast to int() to avoid NumPy int32 overflow
            ntok_total += int(shard_ntok) 
        self.ntok_total = ntok_total

        print(f"DataLoader: total tokens = {ntok_total:,} across {len(self.files)} files")

        self.current_shard = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1
        
        # If we don't have enough tokens in the current shard for a full batch
        if self.current_position + needed > len(self.tokens):
            # 1. Grab what's left in the current shard
            part1 = self.tokens[self.current_position:]
            
            # 2. Advance to the next shard
            self.current_shard = (self.current_shard + 1) % len(self.files)
            self.tokens = _load_data_shard(self.files[self.current_shard])
            
            # 3. Take the remaining needed tokens from the start of the new shard
            # We reset position to 0 because we are starting fresh
            remainder_needed = needed - len(part1)
            part2 = self.tokens[:remainder_needed]
            
            # 4. Concatenate them
            buf = np.concatenate([part1, part2])
            self.current_position = remainder_needed - 1 # -1 because the last token of this batch is the first of the next
        else:
            # Standard case: just slice the current shard
            buf = self.tokens[self.current_position : self.current_position + needed]
            self.current_position += B * T

        # Convert to tensor efficiently
        # Using .astype(np.int64) before torch.from_numpy is generally fastest for uint16 -> long
        pt_buf = torch.from_numpy(buf.astype(np.int64)).to(self.device)
        
        x = pt_buf[:-1].view(B, T)
        y = pt_buf[1:].view(B, T)
        
        return x, y

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
