import math
import time
import torch
import torch.amp
import torch._inductor.config as torch_config

class Trainer:
    def __init__(self, config, logger, device, use_amp):
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
        self.step = 0
        
        self.device = device
    
        self.use_amp = use_amp
        if use_amp:
            self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        if config['hardware']["coordinate_descent_tuning"]:
            if hasattr(torch_config, "coordinate_descent_tuning"):
                torch_config.coordinate_descent_tuning = True

        self.tokens_per_iter  =  S * B * self.GRAD_ACCUMULATION_STEPS

    def get_lr(self, it):
        assert it <= self.NUM_ITERATIONS
        if it < self.WARMUP_ITERS:
            return self.LEARNING_RATE * (it + 1) / self.WARMUP_ITERS
        elif it < self.NUM_ITERATIONS - self.WARMDOWN_ITERS:
            return self.LEARNING_RATE
        else:
            decay_ratio = (self.NUM_ITERATIONS - it) / self.WARMDOWN_ITERS
            return self.LEARNING_RATE * decay_ratio


    def validate(self, model, val_loader, val_steps):
        torch.cuda.synchronize()
        model.eval()
        if val_loader:  # Only if validation loader exists
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()  # ← .item() to avoid GPU memory accumulation
                val_loss /= val_steps

            # Compute perplexity safely
            # Clamp loss to avoid overflow in exp()
            clamped_loss = min(val_loss, 700)  # exp(700) ~ 1e304, near float64 limit
            val_ppl = math.exp(clamped_loss)

            # Log both loss and perplexity
            self.logger.log(
                val_loss=val_loss,
                val_ppl=val_ppl,
                tokens_progres=self.step * self.tokens_per_iter
            )
                

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
            self.step = step  # critical: update current step
            last_step = (step == self.NUM_ITERATIONS)

            # --- Validation (if scheduled) ---
            if self.VAL_LOSS_EVERY > 0 and (step % self.VAL_LOSS_EVERY == 0 or last_step):
                torch.cuda.synchronize()
                t_val_start = time.perf_counter()
                self.validate(model, val_loader, val_steps)
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

            for micro_step in range(self.GRAD_ACCUMULATION_STEPS):
                x, y = train_loader.next_batch()

                if self.use_amp:
                    with self.ctx:
                        _, loss = model(x, y, return_logits=False)
                        loss = loss / self.GRAD_ACCUMULATION_STEPS
                else:
                    _, loss = model(x, y, return_logits=False)
                    loss = loss / self.GRAD_ACCUMULATION_STEPS

                train_loss += loss.detach()
                loss.backward()

            # --- Optimizer step ---
            lr = self.get_lr(self.step)
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
            avg_step_time_ms = total_training_time_ms / (step + 1) if step >= 0 else 0

            if self.VAL_LOSS_EVERY > 0 and (step % self.VAL_LOSS_EVERY == 0 or last_step):
                if total_loss_value > 0:    
                    self.logger.log(
                        step=step,
                        loss=total_loss_value/self.VAL_LOSS_EVERY ,
                        train_time_sec=total_training_time_ms / 1000,
                        step_avg_time_sec=avg_step_time_ms / 1000,
                        val_time_sec=total_validation_time_ms / 1000,
                    )

            if self.logger.check_save_step(step):
                self.logger.save_model(model, step)

        # --- Final summary ---
        self.logger.info(f"We are done SIR!")
      