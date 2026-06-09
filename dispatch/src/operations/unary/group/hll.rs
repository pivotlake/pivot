//! A small HyperLogLog for estimating the distinct group count, used to size the
//! radix merge targets. Standard estimator (Flajolet, Fusy, Gandouet & Meunier,
//! "HyperLogLog: the analysis of a near-optimal cardinality estimation
//! algorithm", 2007) with the linear-counting small-range correction. Inputs are
//! 64-bit hashes, so the paper's large-range (32-bit collision) correction is
//! unnecessary.

/// log2 of the register count. 2^12 = 4096 registers (4 KB) — ~1.6% standard
/// error, far tighter than picking a power-of-two table size needs.
const P: u32 = 12;
const M: usize = 1 << P;

/// HyperLogLog's bias-correction constant α_m. For m ≥ 128 registers the paper
/// gives `α_m = 0.7213 / (1 + 1.079/m)`; `0.7213` and `1.079` are its
/// empirically-fitted values, not tunables.
const ALPHA: f64 = 0.7213 / (1.0 + 1.079 / M as f64);

/// Raw-estimate cutoff, in units of m: below `E = 5/2·m` the harmonic-mean
/// estimator is biased low, so HyperLogLog falls back to linear counting.
const LINEAR_COUNTING_CUTOFF: f64 = 2.5;

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

    /// Fold a row hash into the sketch: its low `P` bits choose a register, and
    /// the position of the leftmost 1-bit in the rest (+1) is recorded as that
    /// register's running max. `hash >> P` has its top `P` bits zero, so
    /// `leading_zeros() >= P` and the rank lands in `1..=64-P+1`.
    #[inline(always)]
    pub fn add(&mut self, hash: u64) {
        let idx = (hash & (M as u64 - 1)) as usize;
        let rank = ((hash >> P).leading_zeros() - P + 1) as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Combine another worker's sketch (register-wise max) into this one.
    pub fn merge(&mut self, other: &Hll) {
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// Estimated distinct count: `α_m · m² / Σ2^-register` (the harmonic-mean
    /// estimator), with linear counting in the small-cardinality regime.
    pub fn estimate(&self) -> usize {
        let m = M as f64;
        let indicator: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0f64.powi(-(r as i32)))
            .sum();
        let raw = ALPHA * m * m / indicator;

        if raw <= LINEAR_COUNTING_CUTOFF * m {
            let empty = self.registers.iter().filter(|&&r| r == 0).count();
            if empty > 0 {
                // Linear counting: m · ln(m / empty_registers).
                return (m * (m / empty as f64).ln()).round() as usize;
            }
        }
        raw.round() as usize
    }
}
