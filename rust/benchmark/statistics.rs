pub struct Stats {
    pub mean: f64,
    pub sd: f64,    // Standard deviation
    pub ci_95: f64, // 95% confidence interval half-width
    pub cv: f64,    // Coefficient of variation (SD/mean as percentage)
}

pub fn calculate_stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    if n < 2.0 {
        let mean = if n > 0.0 { values[0] } else { 0.0 };
        return Stats {
            mean,
            sd: 0.0,
            ci_95: 0.0,
            cv: 0.0,
        };
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = variance.sqrt();
    let std_error = sd / n.sqrt();
    let ci_95 = Z_95 * std_error;
    let cv = if mean > 0.0 { (sd / mean) * 100.0 } else { 0.0 };
    Stats {
        mean,
        sd,
        ci_95,
        cv,
    }
}

pub fn calculate_stats_u64(values: &[u64]) -> Stats {
    let floats: Vec<f64> = values.iter().map(|&x| x as f64).collect();
    calculate_stats(&floats)
}

const Z_95: f64 = 1.96; // Z-score for 95% confidence interval
