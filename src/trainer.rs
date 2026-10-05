use candle_core::Result;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

pub struct CosineAnnealingLr {
    pub base_lr: f64,
    pub total_steps: usize,
    pub min_lr_ratio: f64,
}

impl CosineAnnealingLr {
    pub fn new(base_lr: f64, total_steps: usize) -> Self {
        Self {
            base_lr,
            total_steps,
            min_lr_ratio: 0.1,
        }
    }

    pub fn get_lr(&self, current_step: usize) -> f64 {
        if self.total_steps == 0 {
            return self.base_lr;
        }
        let step = current_step.min(self.total_steps);
        let progress = step as f64 / self.total_steps as f64;
        let pi = std::f64::consts::PI;
        self.base_lr * (self.min_lr_ratio + 0.45 * (1.0 + (pi * progress).cos()))
    }
}

pub struct Trainer {
    pub varmap: VarMap,
    pub optimizer: AdamW,
    pub lr_scheduler: CosineAnnealingLr,
    pub grad_clip: Option<f64>,
    pub current_step: usize,
}

impl Trainer {
    pub fn new(
        varmap: VarMap,
        learning_rate: f64,
        total_steps: usize,
        grad_clip: Option<f64>,
    ) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;
        let lr_scheduler = CosineAnnealingLr::new(learning_rate, total_steps);

        Ok(Self {
            varmap,
            optimizer,
            lr_scheduler,
            grad_clip,
            current_step: 0,
        })
    }

    pub fn step_optimizer(&mut self, loss: &candle_core::Tensor) -> Result<()> {
        let lr = self.lr_scheduler.get_lr(self.current_step);
        self.optimizer.set_learning_rate(lr);

        let grads = loss.backward()?;
        self.optimizer.step(&grads)?;

        self.current_step += 1;
        Ok(())
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)
    }
}
