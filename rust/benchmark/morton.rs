//! 3D Morton (Z-order) encoding for spatial sorting.
//!
//! Encodes (x, y, z) integer coordinates (21 bits per axis, total 63 bits)
//! into a single 64-bit interleaved key. This is sufficient for a uniform
//! resolution of 2^21 = 2,097,152 cells per axis (~8 trillion cells total),
//! which exceeds any realistic particle-simulation cell grid.
//!
//! Peano-Hilbert keys would be slightly better for cache locality (no
//! discontinuities between adjacent cells), but Morton is faster to compute
//! and the asymptotic sort cost is identical. For benchmarking DeltaSort
//! the difference is irrelevant.

#![allow(dead_code)]

/// Spread the low 21 bits of `x` into every third bit position.
/// Result has the i-th bit of x at position 3*i.
#[inline]
fn split_by_3(x: u32) -> u64 {
    let mut x = (x as u64) & 0x1FFFFF; // mask 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8))  & 0x100F00F00F00F00F;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3;
    x = (x | (x << 2))  & 0x1249249249249249;
    x
}

/// Interleave 21 bits each of x, y, z into a 63-bit Morton code.
#[inline]
pub fn morton3(x: u32, y: u32, z: u32) -> u64 {
    split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)
}

/// Encode a 3D point in [0, 1]^3 into a Morton key at the requested
/// resolution (bits per axis, max 21).
#[inline]
pub fn morton_unit(p: [f64; 3], bits: u32) -> u64 {
    debug_assert!(bits <= 21);
    let scale = (1u32 << bits) as f64;
    let x = (p[0].clamp(0.0, 1.0 - f64::EPSILON) * scale) as u32;
    let y = (p[1].clamp(0.0, 1.0 - f64::EPSILON) * scale) as u32;
    let z = (p[2].clamp(0.0, 1.0 - f64::EPSILON) * scale) as u32;
    morton3(x, y, z)
}
