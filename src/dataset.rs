use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftSample {
    pub conversations: Vec<ConversationMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainSample {
    pub text: String,
}

pub struct PretrainDataset {
    pub samples: Vec<String>,
    pub tokenizer: Tokenizer,
    pub max_length: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl PretrainDataset {
    pub fn from_file(
        path: &str,
        tokenizer: Tokenizer,
        max_length: usize,
        bos_token_id: u32,
        eos_token_id: u32,
        pad_token_id: u32,
    ) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Failed to open pretrain data: {}", path))?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(sample) = serde_json::from_str::<PretrainSample>(trimmed) {
                samples.push(sample.text);
            } else {
                samples.push(trimmed.to_string());
            }
        }

        Ok(Self {
            samples,
            tokenizer,
            max_length,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get_batch(
        &self,
        indices: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let mut input_ids_batch = Vec::with_capacity(indices.len() * self.max_length);
        let mut labels_batch = Vec::with_capacity(indices.len() * self.max_length);

        for &idx in indices {
            let sample_idx = idx % self.samples.len();
            let text = &self.samples[sample_idx];

            let encoding = self
                .tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
            let mut ids = encoding.get_ids().to_vec();

            if ids.len() > self.max_length.saturating_sub(2) {
                ids.truncate(self.max_length - 2);
            }

            let mut full_ids = Vec::with_capacity(self.max_length);
            full_ids.push(self.bos_token_id);
            full_ids.extend_from_slice(&ids);
            full_ids.push(self.eos_token_id);

            let pad_count = self.max_length.saturating_sub(full_ids.len());
            for _ in 0..pad_count {
                full_ids.push(self.pad_token_id);
            }

            for (i, &tok) in full_ids.iter().enumerate() {
                input_ids_batch.push(tok as i64);
                if i >= full_ids.len() - pad_count {
                    labels_batch.push(-100i64);
                } else {
                    labels_batch.push(tok as i64);
                }
            }
        }

        let input_tensor = Tensor::from_vec(
            input_ids_batch,
            (indices.len(), self.max_length),
            device,
        )?;
        let labels_tensor = Tensor::from_vec(
            labels_batch,
            (indices.len(), self.max_length),
            device,
        )?;

        Ok((input_tensor, labels_tensor))
    }
}

pub struct SftDataset {
    pub samples: Vec<SftSample>,
    pub tokenizer: Tokenizer,
    pub max_length: usize,
    pub pad_token_id: u32,
}

impl SftDataset {
    pub fn from_file(
        path: &str,
        tokenizer: Tokenizer,
        max_length: usize,
        pad_token_id: u32,
    ) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Failed to open SFT data: {}", path))?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(sample) = serde_json::from_str::<SftSample>(trimmed) {
                samples.push(sample);
            }
        }

        Ok(Self {
            samples,
            tokenizer,
            max_length,
            pad_token_id,
        })
    }

    pub fn format_prompt(messages: &[ConversationMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
        prompt
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
