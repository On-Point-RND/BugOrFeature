import math
import time
import torch
import torch.amp
import torch._inductor.config as torch_config

from custom_layers.base_auxiliary_loss import collect_auxiliary_losses
from validation_prompt import SAMPLE_SPECT

try:
    import tiktoken  # for GPT-2 encode/decode (optional)
except Exception:  # pragma: no cover
    tiktoken = None


class Trainer:
    def __init__(self, config, logger, last_step, device, use_amp, total_dataset_tokens=None):
        self.NUM_ITERATIONS = config['training']['num_iterations']
        self.VAL_LOSS_EVERY = config['evaluation']['val_loss_every']
        self.WARMUP_ITERS = config['training']['warmup_iters']
        self.LEARNING_RATE = config['training']['learning_rate']
        self.WARMDOWN_ITERS = config['training']['warmdown_iters']
        self.torch_compile = config['hardware']['compile']

        S = config['training']['sequence_length']
        B = config['training']['batch_size']

        self.GRAD_ACCUMULATION_STEPS = config['training']['grad_accumulation_steps']
        self.logger = logger
        self.step = last_step

        self.device = device

        self.use_amp = use_amp
        if use_amp:
            self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        if config['hardware']["coordinate_descent_tuning"]:
            if hasattr(torch_config, "coordinate_descent_tuning"):
                torch_config.coordinate_descent_tuning = True

        self.tokens_per_iter = S * B * self.GRAD_ACCUMULATION_STEPS
        self.total_dataset_tokens = total_dataset_tokens
        
        # Log dataset info
        if self.total_dataset_tokens:
            self.logger.info(f"Total tokens in dataset: {self.total_dataset_tokens:,}")
            max_tokens = self.NUM_ITERATIONS * self.tokens_per_iter
            self.logger.info(f"Max tokens to process (all iterations): {max_tokens:,}")
        

    def _maybe_decode_gpt2(self, token_ids: torch.Tensor):
        """Try to decode GPT-2 token ids into text using tiktoken. Return None if unavailable."""
        if tiktoken is None:
            return None
        try:
            enc = tiktoken.get_encoding("gpt2")
            if token_ids.dim() == 2:
                token_ids = token_ids[0]
            return enc.decode(list(map(int, token_ids.detach().cpu().tolist())))
        except Exception:
            return None

    def _encode_prompt(self, prompt: str | None, bos_token: int = 50256) -> torch.Tensor:
        """Encode prompt to [1, T] on device. Fallback to BOS when tiktoken/prompt is missing."""
        if prompt and tiktoken is not None:
            try:
                enc = tiktoken.get_encoding("gpt2")
                ids = enc.encode(prompt)
                if not ids:
                    ids = [bos_token]
                return torch.tensor([ids], dtype=torch.long, device=self.device)
            except Exception:
                pass
        return torch.tensor([[bos_token]], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _generate_tokens(
        self,
        model,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        prompt: str | None = None,
        bos_token: int = 50256,
    ) -> torch.Tensor:
        """Greedy/top-k generation starting from prompt (or BOS). Returns shape [T_total]."""
        model.eval()
        idx = self._encode_prompt(prompt, bos_token=bos_token)  # [1, T0]
        with (self.ctx if getattr(self, 'ctx', None) is not None else torch.no_grad()):
            for _ in range(max_new_tokens):
                logits, _ = model(idx, None, return_logits=True)  # [1, T, vocab]
                logits = logits[:, -1, :]  # [1, vocab]
                if temperature != 1.0:
                    logits = logits / max(1e-8, float(temperature))
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    thresh = v[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # greedy
                idx = torch.cat((idx, next_token), dim=1)
        return idx[0]

    def get_lr(self, it):
        assert it <= self.NUM_ITERATIONS
        if it < self.WARMUP_ITERS:
            return self.LEARNING_RATE * (it + 1) / self.WARMUP_ITERS
        elif it < self.NUM_ITERATIONS - self.WARMDOWN_ITERS:
            return self.LEARNING_RATE
        else:
            decay_ratio = (self.NUM_ITERATIONS - it) / self.WARMDOWN_ITERS
            return self.LEARNING_RATE * decay_ratio

    def validate(self, model, val_loader, val_steps, step):
        torch.cuda.synchronize()
        model.eval()
        if val_loader:  # Only if validation loader exists
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= val_steps

            # Perplexity
            clamped_loss = min(val_loss, 700)
            val_ppl = math.exp(clamped_loss)

            log_dict = {
                "step":step,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "tokens_processed": step * self.tokens_per_iter,
            }
            
            if self.total_dataset_tokens:
                dataset_progress_pct = (step * self.tokens_per_iter / self.total_dataset_tokens) * 100
                log_dict["dataset_progress_pct"] = dataset_progress_pct
            
            self.logger.log(**log_dict)

            self.logger.collect_and_log_metadata(model,step, self.tokens_per_iter)

            try:
                torch.manual_seed(1234)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(1234)
            except Exception:
                pass


            samples_out = []
            for spec in SAMPLE_SPECT:
                tok_seq = self._generate_tokens(
                    model,
                    max_new_tokens=spec["gen"]["max_new_tokens"],
                    temperature=spec["gen"]["temperature"],
                    top_k=spec["gen"]["top_k"],
                    prompt=spec["prompt"],
                    bos_token=50256,
                )
                text = self._maybe_decode_gpt2(tok_seq)
                samples_out.append({
                    "name": spec["name"],
                    "prompt": spec["prompt"],
                    "generation_params": spec["gen"],
                    "text": text,
                })

            payload = {
                "step": int(self.step),
                "tokens_per_iter": int(self.tokens_per_iter),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "samples": samples_out,
            }
            self.logger.save_sample_artifact(payload, step=step, multi=True)

    def train(self, train_loader, val_loader, model, val_steps):
        # --- Optional model compilation ---
        if self.torch_compile:
            self.logger.info("Started model compilation")
            compile_start = time.perf_counter()
            model = torch.compile(model)  # NOTE: may cause issues on some GPUs
            compile_time = time.perf_counter() - compile_start
            self.logger.info(f"Model compiled in {compile_time:.2f} seconds")

        # --- Initialize timing variables ---
        total_training_time_ms = 0.0
        total_validation_time_ms = 0.0

        self.logger.info("Training has started ... wait and relax ... 🦫")
        total_loss_value = 0
        for step in range(self.NUM_ITERATIONS + 1): 
            total_step = step+self.step
            last_step = (total_step == self.NUM_ITERATIONS)

            # --- Validation (if scheduled) ---
            if self.VAL_LOSS_EVERY > 0 and (step % self.VAL_LOSS_EVERY == 0 or last_step):
                torch.cuda.synchronize()
                t_val_start = time.perf_counter()
                self.validate(model, val_loader, val_steps, total_step)
                torch.cuda.synchronize()
                val_time_ms = 1000 * (time.perf_counter() - t_val_start)
                total_validation_time_ms += val_time_ms

            if last_step:
                break

            # --- Training step ---
            torch.cuda.synchronize()
            t_train_start = time.perf_counter()

            model.train()
            train_loss = torch.zeros(1, device=self.device)

            for _ in range(self.GRAD_ACCUMULATION_STEPS):
                x, y = train_loader.next_batch()

                if self.use_amp:
                    with self.ctx:
                        _, loss = model(x, y, return_logits=False)
                        loss = loss / self.GRAD_ACCUMULATION_STEPS
                else:
                    _, loss = model(x, y, return_logits=False)
                    loss = loss / self.GRAD_ACCUMULATION_STEPS

               
                aux_loss = collect_auxiliary_losses(model, weighted=True, return_details=False)
                if aux_loss.item() != 0:
                    aux_loss = aux_loss / self.GRAD_ACCUMULATION_STEPS
                    loss = loss + aux_loss
                
                train_loss += loss.detach()
                loss.backward()

            # --- Optimizer step ---
            lr = self.get_lr(total_step)
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = lr

            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            # --- Record training time for this step ---
            torch.cuda.synchronize()
            step_train_time_ms = 1000 * (time.perf_counter() - t_train_start)
            total_training_time_ms += step_train_time_ms

            # --- Logging ---
            total_loss_value += train_loss.item()
            avg_step_time_ms = total_training_time_ms / (total_step + 1) if total_step >= 0 else 0

            if self.VAL_LOSS_EVERY > 0 and (total_step % self.VAL_LOSS_EVERY == 0 or last_step):
                # Calculate token progress (after training step, so step+1)
                if train_loss > 0:
                    tokens_processed = 100*(step+1) * self.tokens_per_iter / self.total_dataset_tokens
                    log_dict = {
                        "step": total_step,
                        "loss": total_loss_value / self.VAL_LOSS_EVERY,
                        "train_time_sec": total_training_time_ms / 1000,
                        "step_avg_time_sec": avg_step_time_ms / 1000,
                        "val_time_sec": total_validation_time_ms / 1000,
                        "tokens_processed_percent": tokens_processed,
                    }

                
                self.logger.info(f'Tokens processed {tokens_progress}%')
                
                self.logger.log(**log_dict)
                total_loss_value = 0  

           
            if self.logger.check_save_step(total_step):
                self.logger.save_model_and_config(model, total_step)

        # --- Final summary ---
        self.logger.info(f"We are done SIR!")
      