//! A small HyperLogLog for sizing radix merge targets.

/// log2 of the register count. 2^12 = 4096 registers (4 KB).
const P: u32 = 12;
const M: usize = 1 << P;

#[derive(Clone)]
pub struct Hll {
    registers: Box<[u8; M]>,
}

impl Default for Hll {
    fn default() -> Self {
        Self::new()
    }
}

impl Hll {
    pub fn new() -> Self {
        Self {
            registers: vec![0u8; M].into_boxed_slice().try_into().unwrap(),
        }
    }

    #[inline(always)]
    pub fn add(&mut self, hash: u64) {
        let idx = (hash & (M as u64 - 1)) as usize;
        let rank = ((hash >> P).leading_zeros() - P + 1) as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    pub fn merge(&mut self, other: &Hll) {
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = (*a).max(*b);
        }
    }

    pub fn estimate(&self) -> usize {
        let m = M as f64;
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0f64.powi(-(r as i32)))
            .sum();
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let raw = alpha * m * m / sum;

        if raw <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return (m * (m / zeros as f64).ln()).round() as usize;
            }
        }
        raw.round() as usize
    }
}
