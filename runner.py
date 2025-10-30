import time
import torch
import yaml

from trainer import Trainer
from logger import Logger
from model import GPT, GPTConfig
from datasets import BaseDataLoader

from swap_layers import apply_simple_linear_swaps

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Logger 
logger = Logger(cfg, use_ml_flow=cfg['logging']['use_ml_flow'])

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

logger.info("==="*10 + f"\nMODEL BEFORE SWAP:\n\n{model}\n\n" + "==="*10 )

apply_simple_linear_swaps(model, cfg, logging=logger.info if cfg['logging'].get('use_ml_flow') else print)

logger.info("\n\n" + "==="*10 + f"\nMODEL AFTER SWAP:\n\n{model}\n\n" + "==="*10 )

model.set_optimizers(
    weight_decay=cfg["training"]["weight_decay"],
    learning_rate=cfg["training"]["learning_rate"],
    betas=cfg["training"]["optimizer_betas"],
)

logger.save_model_architecture(model)

# Warmup the data pipeline
x, y = train_loader.next_batch()

# Timing
torch.cuda.synchronize()

#  Trainer
trainer = Trainer(
    cfg, 
    logger, 
    cfg['hardware']["device"], 
    use_amp=cfg['hardware']['amp'],
    total_dataset_tokens=train_loader.ntok_total
)

trainer.train(train_loader, val_loader, model, val_steps)