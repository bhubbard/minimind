use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct LayerCache {
    pub key: Option<Tensor>,
    pub value: Option<Tensor>,
}

impl LayerCache {
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }

    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.key, &self.value) {
            (Some(prev_k), Some(prev_v)) => {
                let new_k = Tensor::cat(&[prev_k, &k], 1)?;
                let new_v = Tensor::cat(&[prev_v, &v], 1)?;
                self.key = Some(new_k.clone());
                self.value = Some(new_v.clone());
                Ok((new_k, new_v))
            }
            _ => {
                self.key = Some(k.clone());
                self.value = Some(v.clone());
                Ok((k, v))
            }
        }
    }

    pub fn current_seq_len(&self) -> usize {
        self.key.as_ref().map(|k| k.dim(1).unwrap_or(0)).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.key = None;
        self.value = None;
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerCache::new());
        }
        Self { layers }
    }

    pub fn current_seq_len(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.current_seq_len())
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            l.reset();
        }
    }
}
