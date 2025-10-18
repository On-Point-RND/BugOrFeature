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


class BaseDataLoader:
    def __init__(self, filename_pattern, B, T, device):
        self.B = B
        self.T = T
        self.device = device

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count total tokens
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # Ensure shard is large enough for at least one full batch + 1 token
            assert shard_ntok >= B * T + 1, f"Shard {fname} too small: {shard_ntok} tokens"
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total tokens = {ntok_total:,} across {len(self.files)} files")

        # initialize
        self.current_shard = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = 0

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # Check if current shard has enough tokens left
        if self.current_position + B * T + 1 > len(self.tokens):
            self.advance()

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long, device=self.device)
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets

        self.current_position += B * T
        return x, y