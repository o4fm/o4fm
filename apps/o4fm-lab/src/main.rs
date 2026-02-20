use std::env;

use o4fm_core::{FecProfile, RadioProfile, SymbolRate};
use o4fm_fec::{decode_ldpc, encode_ldpc};
use o4fm_phy::{demodulate, modulate};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

#[derive(Debug, Clone)]
struct Config {
    trials: usize,
    payload_bits: usize,
    noise_start: f32,
    noise_end: f32,
    noise_step: f32,
    burst_prob: f32,
    burst_len: usize,
    burst_amp: f32,
    gain: f32,
    dc_offset: f32,
    clip: f32,
    symbol_rate: SymbolRate,
    max_iterations: u8,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            trials: 100,
            payload_bits: 128,
            noise_start: 100.0,
            noise_end: 1000.0,
            noise_step: 100.0,
            burst_prob: 0.0,
            burst_len: 0,
            burst_amp: 2500.0,
            gain: 1.0,
            dc_offset: 0.0,
            clip: f32::from(i16::MAX),
            symbol_rate: SymbolRate::R4800,
            max_iterations: 16,
            seed: 0x0F4F_2026,
        }
    }
}

#[derive(Default)]
struct Stats {
    total_bits: usize,
    bit_errors: usize,
    frames: usize,
    frame_errors: usize,
    decode_failures: usize,
    snr_acc: f32,
}

fn main() {
    let cfg = parse_args(env::args().skip(1).collect());
    let mut radio = RadioProfile::default();
    radio.symbol_rate = cfg.symbol_rate;

    let mut fec = FecProfile::default();
    fec.code_k = u16::try_from(cfg.payload_bits).expect("payload bits must fit u16");
    fec.code_n = fec.code_k.saturating_mul(2);
    fec.max_iterations = cfg.max_iterations;

    let mut rng = StdRng::seed_from_u64(cfg.seed);

    println!("o4fm-lab stress sweep");
    println!(
        "  cfg: trials={} payload_bits={} noise=[{:.1}..{:.1}] step={:.1} symbol_rate={} max_iters={}",
        cfg.trials,
        cfg.payload_bits,
        cfg.noise_start,
        cfg.noise_end,
        cfg.noise_step,
        radio.symbol_rate.as_hz(),
        cfg.max_iterations
    );
    println!(
        "  channel: gain={:.2} dc_offset={:.1} clip={:.1} burst_prob={:.4} burst_len={} burst_amp={:.1}",
        cfg.gain, cfg.dc_offset, cfg.clip, cfg.burst_prob, cfg.burst_len, cfg.burst_amp
    );
    println!("  columns: noise_std, ber, fer, decode_fail_rate, avg_snr_db");

    for noise in noise_points(&cfg) {
        let mut stats = Stats::default();
        for _ in 0..cfg.trials {
            run_trial(&cfg, &radio, &fec, noise, &mut rng, &mut stats);
        }

        let ber = ratio(stats.bit_errors, stats.total_bits);
        let fer = ratio(stats.frame_errors, stats.frames);
        let dfr = ratio(stats.decode_failures, stats.frames);
        let snr = if stats.frames > 0 {
            stats.snr_acc / stats.frames as f32
        } else {
            0.0
        };

        println!("  {noise:8.1}, {ber:0.6}, {fer:0.6}, {dfr:0.6}, {snr:0.2}");
    }
}

fn run_trial(
    cfg: &Config,
    radio: &RadioProfile,
    fec: &FecProfile,
    noise_std: f32,
    rng: &mut StdRng,
    stats: &mut Stats,
) {
    let payload_bits: Vec<u8> = (0..cfg.payload_bits)
        .map(|_| u8::from(rng.random::<bool>()))
        .collect();

    let encoded = encode_ldpc(&payload_bits, fec);
    let pcm = modulate(&encoded, radio).expect("modulate must succeed");

    let mut channel = add_awgn(&pcm, noise_std, rng);
    apply_channel_impairments(&mut channel, cfg, rng);

    let demod = demodulate(&channel, radio).expect("demodulate must succeed");

    stats.frames += 1;
    stats.snr_acc += demod.snr_est;

    match decode_ldpc(&demod.soft_bits, fec) {
        Ok(decoded) => {
            let raw_errors = decoded
                .iter()
                .zip(payload_bits.iter())
                .filter(|(a, b)| a != b)
                .count();
            let inv_errors = decoded
                .iter()
                .zip(payload_bits.iter())
                .filter(|(a, b)| (**a ^ 1) != **b)
                .count();
            let best = raw_errors.min(inv_errors);
            stats.total_bits += payload_bits.len();
            stats.bit_errors += best;
            if best > 0 {
                stats.frame_errors += 1;
            }
        }
        Err(_) => {
            stats.decode_failures += 1;
            stats.frame_errors += 1;
            stats.total_bits += payload_bits.len();
            stats.bit_errors += payload_bits.len();
        }
    }
}

fn parse_args(args: Vec<String>) -> Config {
    let mut cfg = Config::default();

    for arg in args {
        if arg == "--help" || arg == "-h" {
            print_help_and_exit();
        }
        let Some((k, v)) = arg.split_once('=') else {
            continue;
        };

        match k {
            "--trials" => cfg.trials = parse_or(v, cfg.trials),
            "--payload-bits" => cfg.payload_bits = parse_or(v, cfg.payload_bits),
            "--noise" => {
                let n = parse_or(v, cfg.noise_start);
                cfg.noise_start = n;
                cfg.noise_end = n;
                cfg.noise_step = 1.0;
            }
            "--noise-start" => cfg.noise_start = parse_or(v, cfg.noise_start),
            "--noise-end" => cfg.noise_end = parse_or(v, cfg.noise_end),
            "--noise-step" => cfg.noise_step = parse_or(v, cfg.noise_step),
            "--burst-prob" => cfg.burst_prob = parse_or(v, cfg.burst_prob),
            "--burst-len" => cfg.burst_len = parse_or(v, cfg.burst_len),
            "--burst-amp" => cfg.burst_amp = parse_or(v, cfg.burst_amp),
            "--gain" => cfg.gain = parse_or(v, cfg.gain),
            "--dc-offset" => cfg.dc_offset = parse_or(v, cfg.dc_offset),
            "--clip" => cfg.clip = parse_or(v, cfg.clip),
            "--symbol-rate" => {}
            "--max-iters" => cfg.max_iterations = parse_or(v, cfg.max_iterations),
            "--seed" => cfg.seed = parse_or(v, cfg.seed),
            _ => {}
        }
    }

    cfg.payload_bits = cfg.payload_bits.clamp(64, 128);
    cfg.noise_step = cfg.noise_step.max(1.0);
    cfg.clip = cfg.clip.max(2000.0);
    cfg.burst_prob = cfg.burst_prob.clamp(0.0, 1.0);

    cfg
}

fn print_help_and_exit() -> ! {
    println!("o4fm-lab options (key=value):");
    println!("  --trials=100");
    println!("  --payload-bits=128           (64..128 for TC256)");
    println!("  --noise=400                  (single point)");
    println!("  --noise-start=100 --noise-end=1000 --noise-step=100");
    println!("  --symbol-rate is fixed at 4800");
    println!("  --max-iters=16");
    println!("  --gain=1.0 --dc-offset=0 --clip=32767");
    println!("  --burst-prob=0.0 --burst-len=0 --burst-amp=2500");
    println!("  --seed=252649510");
    std::process::exit(0);
}

fn parse_or<T: std::str::FromStr>(s: &str, default: T) -> T {
    s.parse().ok().unwrap_or(default)
}

fn noise_points(cfg: &Config) -> Vec<f32> {
    if cfg.noise_start >= cfg.noise_end {
        return vec![cfg.noise_start];
    }

    let mut out = Vec::new();
    let mut n = cfg.noise_start;
    while n <= cfg.noise_end + 0.1 {
        out.push(n);
        n += cfg.noise_step;
    }
    out
}

fn ratio(num: usize, den: usize) -> f32 {
    if den == 0 {
        0.0
    } else {
        num as f32 / den as f32
    }
}

fn add_awgn(samples: &[i16], std_dev: f32, rng: &mut impl Rng) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| {
            let noise = box_muller(rng) * std_dev;
            let v = f32::from(s) + noise;
            v.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
        })
        .collect()
}

fn apply_channel_impairments(samples: &mut [i16], cfg: &Config, rng: &mut impl Rng) {
    let mut burst_remaining = 0usize;

    for sample in samples {
        let mut x = f32::from(*sample);

        if cfg.burst_prob > 0.0 && burst_remaining == 0 && rng.random::<f32>() < cfg.burst_prob {
            burst_remaining = cfg.burst_len;
        }

        if burst_remaining > 0 {
            x += (rng.random::<f32>() * 2.0 - 1.0) * cfg.burst_amp;
            burst_remaining -= 1;
        }

        x = x * cfg.gain + cfg.dc_offset;
        x = x.clamp(-cfg.clip, cfg.clip);
        *sample = x.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
    }
}

fn box_muller(rng: &mut impl Rng) -> f32 {
    let u1 = rng.random::<f32>().max(1e-6);
    let u2 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * core::f32::consts::PI * u2).cos()
}
