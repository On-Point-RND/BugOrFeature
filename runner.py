import time
import torch
import yaml

from trainer import Trainer
from logger import Logger
from model import GPT, GPTConfig
from datasets import BaseDataLoader

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Build model config correctly
model_cfg = GPTConfig(
    vocab_size=cfg["model"]["vocab_size"],
    n_layer=cfg["model"]["n_layer"],
    n_head=cfg["model"]["n_head"],
    n_embd=cfg["model"]["n_embed"]
)

# Data loaders
B = cfg["training"]["batch_size"]
T = cfg["training"]["sequence_length"]

train_loader = BaseDataLoader(cfg["data"]["input_bin"], B, T, cfg['hardware']["device"])

val_loader = val_steps = None
if cfg["evaluation"]["val_loss_every"] > 0:
    val_tokens = cfg["evaluation"]["val_tokens"]
    val_batch_size = cfg["evaluation"]["val_batch_size"]
    tokens_per_iter_val = val_batch_size * T
    assert val_tokens % tokens_per_iter_val == 0, "VAL_TOKENS must be divisible by tokens_per_iter_val"
    val_steps = val_tokens // tokens_per_iter_val
    val_loader = BaseDataLoader(cfg["data"]["input_val_bin"], val_batch_size, T, cfg['hardware']["device"])

# Model
model = GPT(model_cfg).train().to(cfg['hardware']["device"])
model.set_optimizers(
    weight_decay=cfg["training"]["weight_decay"],
    learning_rate=cfg["training"]["learning_rate"],
    betas=cfg["training"]["optimizer_betas"],
)

# Warmup the data pipeline
x, y = train_loader.next_batch()

# Timing
torch.cuda.synchronize()

# Logger & Trainer
logger = Logger(cfg, use_ml_flow=cfg['logging']['use_ml_flow'])
trainer = Trainer(cfg, logger, cfg['hardware']["device"], use_amp=cfg['hardware']['amp'])

trainer.train(train_loader, val_loader, model, val_steps)