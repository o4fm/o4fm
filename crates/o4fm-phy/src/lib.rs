use core::f32::consts::PI;

use o4fm_core::RadioProfile;
use thiserror::Error;

pub const AUDIO_SAMPLE_RATE_HZ: u32 = 48_000;
const CARRIER_CENTER_HZ: f32 = 6000.0;
const CHANNEL_BW_HZ: f32 = 10_000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingEstimate {
    pub samples_per_symbol: usize,
    pub symbol_phase: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DemodResult {
    pub soft_bits: Vec<f32>,
    pub timing: TimingEstimate,
    pub freq_offset: f32,
    pub snr_est: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PhyError {
    #[error("modulation not supported")]
    UnsupportedModulation,
    #[error("invalid symbol rate")]
    InvalidSymbolRate,
    #[error("invalid modulation/rate for channel bandwidth")]
    InvalidProfile,
}

pub fn modulate(frame_bits: &[u8], profile: &RadioProfile) -> Result<Vec<i16>, PhyError> {
    let sps = samples_per_symbol(profile)?;
    let tones = tone_frequencies(profile)?;
    let bits_per_symbol = profile.modulation.bits_per_symbol();
    let symbols = frame_bits.len().div_ceil(bits_per_symbol);

    let mut pcm = Vec::<i16>::with_capacity(symbols * sps);
    let mut phase = 0f32;
    for sym in 0..symbols {
        let symbol = bits_to_symbol(frame_bits, sym * bits_per_symbol, bits_per_symbol);
        let symbol_idx = usize::from(bin_to_gray(symbol)).min(tones.len().saturating_sub(1));
        let freq = tones[symbol_idx];
        let dphi = 2.0 * PI * freq / AUDIO_SAMPLE_RATE_HZ as f32;

        for _ in 0..sps {
            phase += dphi;
            if phase > 2.0 * PI {
                phase -= 2.0 * PI;
            }
            let sample = (phase.sin() * 0.7 * f32::from(i16::MAX)) as i16;
            pcm.push(sample);
        }
    }

    Ok(pcm)
}

pub fn demodulate(pcm_i16: &[i16], profile: &RadioProfile) -> Result<DemodResult, PhyError> {
    let sps = samples_per_symbol(profile)?;
    let tones = tone_frequencies(profile)?;
    let bits_per_symbol = profile.modulation.bits_per_symbol();
    if pcm_i16.is_empty() {
        return Ok(DemodResult {
            soft_bits: Vec::new(),
            timing: TimingEstimate {
                samples_per_symbol: sps,
                symbol_phase: 0.0,
            },
            freq_offset: 0.0,
            snr_est: 0.0,
        });
    }

    let mut pcm: Vec<f32> = pcm_i16.iter().map(|x| f32::from(*x)).collect();
    agc_normalize(&mut pcm);

    let symbol_count = pcm.len() / sps;
    let mut soft_bits = Vec::with_capacity(symbol_count * bits_per_symbol);
    let mut signal_energy = 0f32;
    let mut noise_energy = 0f32;

    for sym in 0..symbol_count {
        let start = sym * sps;
        let end = start + sps;
        let slice = &pcm[start..end];
        let mut energies = vec![0.0_f32; tones.len()];
        let mut best = f32::MIN;
        let mut second = f32::MIN;
        for (idx, tone) in tones.iter().enumerate() {
            let e = tone_energy(slice, *tone, AUDIO_SAMPLE_RATE_HZ as f32);
            energies[idx] = e;
            if e > best {
                second = best;
                best = e;
            } else if e > second {
                second = e;
            }
        }
        signal_energy += best;
        noise_energy += second.max(1e-9);

        // Bit-wise soft metrics: max energy among tones with bit=1 vs bit=0.
        // This is noticeably more robust for higher-order FSK than reusing a
        // single symbol confidence for all bits.
        for bit in 0..bits_per_symbol {
            let shift = bits_per_symbol - 1 - bit;
            let mut best_one = f32::MIN;
            let mut best_zero = f32::MIN;
            for (idx, e) in energies.iter().enumerate() {
                let bin_sym = gray_to_bin(idx as u16);
                if ((bin_sym >> shift) & 1) == 1 {
                    if *e > best_one {
                        best_one = *e;
                    }
                } else if *e > best_zero {
                    best_zero = *e;
                }
            }
            soft_bits.push((best_one - best_zero).clamp(-1.0e6, 1.0e6));
        }
    }

    let snr_est = if noise_energy > 0.0 {
        10.0 * (signal_energy / noise_energy).log10()
    } else {
        99.0
    };
    let freq_offset = estimate_freq_offset(&pcm, AUDIO_SAMPLE_RATE_HZ as f32);

    Ok(DemodResult {
        soft_bits,
        timing: TimingEstimate {
            samples_per_symbol: sps,
            symbol_phase: 0.0,
        },
        freq_offset,
        snr_est,
    })
}

fn bits_to_symbol(bits: &[u8], start: usize, bits_per_symbol: usize) -> u16 {
    let mut out = 0u16;
    for i in 0..bits_per_symbol {
        out <<= 1;
        let bit = bits.get(start + i).copied().unwrap_or(0) & 1;
        out |= u16::from(bit);
    }
    out
}

fn bin_to_gray(v: u16) -> u16 {
    v ^ (v >> 1)
}

fn gray_to_bin(mut g: u16) -> u16 {
    let mut b = g;
    while g > 0 {
        g >>= 1;
        b ^= g;
    }
    b
}

fn tone_frequencies(profile: &RadioProfile) -> Result<Vec<f32>, PhyError> {
    let m = usize::from(profile.modulation.order());
    if m < 2 {
        return Err(PhyError::UnsupportedModulation);
    }

    // Use negotiated tone plan when provided, otherwise fall back to implementation defaults.
    let center_hz = if profile.tone_center_hz == 0 {
        CARRIER_CENTER_HZ
    } else {
        f32::from(profile.tone_center_hz)
    };
    let required_step = if profile.tone_spacing_hz == 0 {
        (profile.symbol_rate.as_hz() as f32 / 2.4).max(250.0)
    } else {
        f32::from(profile.tone_spacing_hz)
    };
    let channel_bw = if profile.rx_bandwidth_hz == 0 {
        CHANNEL_BW_HZ
    } else {
        f32::from(profile.rx_bandwidth_hz)
    };

    let span = required_step * (m as f32 - 1.0);
    let occupied_bw = required_step * m as f32;
    if occupied_bw > channel_bw {
        return Err(PhyError::InvalidProfile);
    }

    let start = center_hz - (span / 2.0);
    if start <= 50.0 || (start + span) >= (AUDIO_SAMPLE_RATE_HZ as f32 / 2.0 - 50.0) {
        return Err(PhyError::InvalidProfile);
    }

    let mut tones = Vec::with_capacity(m);
    for i in 0..m {
        tones.push(start + i as f32 * required_step);
    }
    Ok(tones)
}

fn samples_per_symbol(profile: &RadioProfile) -> Result<usize, PhyError> {
    let symbol_rate = profile.symbol_rate.as_hz();
    if symbol_rate == 0 || !AUDIO_SAMPLE_RATE_HZ.is_multiple_of(symbol_rate) {
        return Err(PhyError::InvalidSymbolRate);
    }
    Ok((AUDIO_SAMPLE_RATE_HZ / symbol_rate) as usize)
}

fn agc_normalize(samples: &mut [f32]) {
    let mut peak = 1.0f32;
    for s in samples.iter() {
        peak = peak.max(s.abs());
    }
    let gain = 1.0 / peak;
    for s in samples {
        *s *= gain;
    }
}

fn tone_energy(samples: &[f32], target_hz: f32, sample_rate_hz: f32) -> f32 {
    let mut i_acc = 0.0f32;
    let mut q_acc = 0.0f32;
    let w = 2.0 * PI * target_hz / sample_rate_hz;
    for (n, &sample) in samples.iter().enumerate() {
        let phase = w * n as f32;
        i_acc += sample * phase.cos();
        q_acc += sample * phase.sin();
    }
    i_acc * i_acc + q_acc * q_acc
}

fn estimate_freq_offset(samples: &[f32], sample_rate_hz: f32) -> f32 {
    if samples.len() < 4 {
        return 0.0;
    }

    let mut crossings = 0usize;
    for pair in samples.windows(2) {
        if (pair[0] <= 0.0 && pair[1] > 0.0) || (pair[0] >= 0.0 && pair[1] < 0.0) {
            crossings += 1;
        }
    }

    let duration_s = samples.len() as f32 / sample_rate_hz;
    let measured_hz = (crossings as f32 / 2.0) / duration_s;
    measured_hz - CARRIER_CENTER_HZ
}

#[cfg(test)]
mod tests {
    use o4fm_core::{Modulation, RadioProfile};

    use super::*;

    #[test]
    fn bfsk_round_trip_soft_bit_sign() {
        let profile = RadioProfile::default();
        let bits = [1u8, 0, 1, 1, 0, 0, 1];
        let pcm = modulate(&bits, &profile).expect("mod ok");
        let out = demodulate(&pcm, &profile).expect("demod ok");
        assert!(out.soft_bits.len() >= bits.len());
        let decoded: Vec<u8> = (0..bits.len())
            .map(|idx| u8::from(out.soft_bits[idx] > 0.0))
            .collect();
        let err = decoded
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| a != b)
            .count();
        let inv_err = decoded
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| (**a ^ 1) != **b)
            .count();
        assert!(
            err <= 1 || inv_err <= 1,
            "too many symbol errors: {err}, inv_err: {inv_err}"
        );
    }

    #[test]
    fn eight_fsk_round_trip_soft_bit_sign() {
        let mut profile = RadioProfile::default();
        profile.modulation = Modulation::EightFsk;
        profile.tone_spacing_hz = 1_000;
        let bits = [0u8, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1];
        let pcm = modulate(&bits, &profile).expect("mod ok");
        let out = demodulate(&pcm, &profile).expect("demod ok");
        assert!(out.soft_bits.len() >= bits.len());
        let decoded: Vec<u8> = (0..bits.len())
            .map(|idx| u8::from(out.soft_bits[idx] > 0.0))
            .collect();
        let err = decoded
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| a != b)
            .count();
        let inv_err = decoded
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| (**a ^ 1) != **b)
            .count();
        assert!(
            err <= 2 || inv_err <= 2,
            "too many 8FSK symbol errors: {err}, inv_err: {inv_err}"
        );
    }

    #[test]
    fn eight_fsk_long_round_trip_error_rate() {
        let mut profile = RadioProfile::default();
        profile.modulation = Modulation::EightFsk;
        profile.tone_spacing_hz = 1_000;
        let bits: Vec<u8> = (0..1536)
            .map(|i| u8::from((i % 2 == 0) || (i % 5 == 0) || (i % 7 == 0)))
            .collect();

        let pcm = modulate(&bits, &profile).expect("mod ok");
        let out = demodulate(&pcm, &profile).expect("demod ok");
        assert!(out.soft_bits.len() >= bits.len());
        let decoded: Vec<u8> = (0..bits.len())
            .map(|idx| u8::from(out.soft_bits[idx] > 0.0))
            .collect();
        let err = decoded
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| a != b)
            .count();
        let inv_err = decoded
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| (**a ^ 1) != **b)
            .count();
        let best_err = err.min(inv_err);
        let ber = best_err as f32 / bits.len() as f32;
        assert!(ber < 0.08, "8FSK BER too high in loopback: {ber:.4}");
    }
}
