use crate::cache::KvCache;
use crate::model::MiniMindForCausalLM;
use candle_core::{Result, Tensor};
use rand::Rng;
use std::collections::HashSet;

pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    pub eos_token_id: u32,
    pub do_sample: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.85,
            top_p: 0.85,
            top_k: 50,
            repetition_penalty: 1.0,
            eos_token_id: 2,
            do_sample: true,
        }
    }
}

pub struct Generator<'a> {
    pub model: &'a MiniMindForCausalLM,
    pub config: GenerationConfig,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a MiniMindForCausalLM, config: GenerationConfig) -> Self {
        Self { model, config }
    }

    pub fn generate<F>(
        &self,
        prompt_tokens: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool, // returns false to stop early
    {
        let device = self.model.device();
        let mut generated = prompt_tokens.to_vec();
        let mut cache = KvCache::new(self.model.config.num_hidden_layers);

        // Pre-fill prompt
        let input_tensor = Tensor::from_vec(
            prompt_tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
            (1, prompt_tokens.len()),
            device,
        )?;

        let mut out = self.model.forward(&input_tensor, Some(&mut cache), None, None)?;

        let mut last_token_logits = out.logits.narrow(1, prompt_tokens.len() - 1, 1)?.squeeze(1)?;

        let mut seen_tokens: HashSet<u32> = prompt_tokens.iter().copied().collect();

        for _ in 0..self.config.max_new_tokens {
            let next_token = self.sample_token(&last_token_logits, &seen_tokens)?;
            generated.push(next_token);
            seen_tokens.insert(next_token);

            // Stream token
            let continue_gen = on_token(next_token);
            if !continue_gen || next_token == self.config.eos_token_id {
                break;
            }

            // Forward next single token with KV cache
            let next_input = Tensor::from_vec(vec![next_token as i64], (1, 1), device)?;
            out = self.model.forward(&next_input, Some(&mut cache), None, None)?;
            last_token_logits = out.logits.narrow(1, 0, 1)?.squeeze(1)?;
        }

        Ok(generated)
    }

    fn sample_token(&self, logits_1d: &Tensor, seen_tokens: &HashSet<u32>) -> Result<u32> {
        let mut logits: Vec<f32> = logits_1d.flatten_all()?.to_vec1()?;

        // Repetition penalty
        if (self.config.repetition_penalty - 1.0).abs() > 1e-5 {
            for &tok in seen_tokens {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.config.repetition_penalty as f32;
                    } else {
                        logits[idx] *= self.config.repetition_penalty as f32;
                    }
                }
            }
        }

        // Temperature
        if self.config.temperature > 1e-5 {
            let temp = self.config.temperature as f32;
            for val in logits.iter_mut() {
                *val /= temp;
            }
        } else {
            // Greedy
            return Ok(argmax(&logits) as u32);
        }

        if !self.config.do_sample {
            return Ok(argmax(&logits) as u32);
        }

        // Top-K
        if self.config.top_k > 0 && self.config.top_k < logits.len() {
            let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = pairs[self.config.top_k - 1].1;
            for val in logits.iter_mut() {
                if *val < threshold {
                    *val = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        let max_val = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let mut sum_exp = 0.0f32;
        let mut exps: Vec<f32> = Vec::with_capacity(logits.len());
        for &l in &logits {
            if l == f32::NEG_INFINITY {
                exps.push(0.0);
            } else {
                let e = (l - max_val).exp();
                exps.push(e);
                sum_exp += e;
            }
        }

        if sum_exp <= 0.0 {
            return Ok(0);
        }

        let mut probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        // Top-P (nucleus)
        if self.config.top_p < 1.0 {
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0f32;
            let mut cutoff_idx = indexed_probs.len();
            for (idx, &(_, p)) in indexed_probs.iter().enumerate() {
                cumsum += p;
                if cumsum > self.config.top_p as f32 {
                    cutoff_idx = idx + 1;
                    break;
                }
            }

            let mut valid_set = HashSet::with_capacity(cutoff_idx);
            for &(orig_idx, _) in &indexed_probs[..cutoff_idx] {
                valid_set.insert(orig_idx);
            }

            let mut re_sum = 0.0f32;
            for i in 0..probs.len() {
                if !valid_set.contains(&i) {
                    probs[i] = 0.0;
                } else {
                    re_sum += probs[i];
                }
            }
            if re_sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= re_sum;
                }
            }
        }

        // Sample categorical
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut acc = 0.0f32;
        for (idx, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                return Ok(idx as u32);
            }
        }

        Ok((probs.len() - 1) as u32)
    }
}

fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
