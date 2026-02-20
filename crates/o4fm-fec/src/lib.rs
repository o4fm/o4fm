#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::trivially_copy_pass_by_ref
)]

use labrador_ldpc::LDPCCode;
use o4fm_core::{FecProfile, FecScheme};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FecError {
    #[error("unsupported FEC scheme")]
    UnsupportedScheme,
    #[error("unsupported LDPC code profile")]
    UnsupportedCodeProfile,
    #[error("invalid length")]
    InvalidLength,
    #[error("decoding failed")]
    DecodingFailed,
}

#[must_use]
pub fn encode_ldpc(payload_bits: &[u8], fec_profile: &FecProfile) -> Vec<u8> {
    if fec_profile.scheme != FecScheme::Ldpc {
        return payload_bits.to_vec();
    }

    let Some(code) = select_code(fec_profile) else {
        return payload_bits.to_vec();
    };

    if payload_bits.len() != code.k() {
        return payload_bits.to_vec();
    }

    let mut codeword = vec![0u8; code.n() / 8];
    let data_bytes = pack_bits(payload_bits);
    code.copy_encode(&data_bytes, &mut codeword);

    let encoded_bits = unpack_bits(&codeword, code.n());
    if fec_profile.interleaver_depth > 1 {
        interleave_block(&encoded_bits, usize::from(fec_profile.interleaver_depth))
    } else {
        encoded_bits
    }
}

pub fn decode_ldpc(llr: &[f32], fec_profile: &FecProfile) -> Result<Vec<u8>, FecError> {
    if fec_profile.scheme != FecScheme::Ldpc {
        return Err(FecError::UnsupportedScheme);
    }

    let code = select_code(fec_profile).ok_or(FecError::UnsupportedCodeProfile)?;
    if llr.len() != code.n() {
        return Err(FecError::InvalidLength);
    }

    let deinterleaved = if fec_profile.interleaver_depth > 1 {
        deinterleave_block(llr, usize::from(fec_profile.interleaver_depth))
    } else {
        llr.to_vec()
    };

    decode_with_polarity(&deinterleaved, code, fec_profile.max_iterations)
        .or_else(|| {
            let inverted: Vec<f32> = deinterleaved.iter().map(|x| -*x).collect();
            decode_with_polarity(&inverted, code, fec_profile.max_iterations)
        })
        .ok_or(FecError::DecodingFailed)
}

fn decode_with_polarity(llr: &[f32], code: LDPCCode, max_iterations: u8) -> Option<Vec<u8>> {
    let quantized = quantize_llr_i8(llr);
    let mut output = vec![0u8; code.output_len()];
    let mut working = vec![0i8; code.decode_ms_working_len()];
    let mut working_u8 = vec![0u8; code.decode_ms_working_u8_len()];

    let (success, _) = code.decode_ms(
        &quantized,
        &mut output,
        &mut working,
        &mut working_u8,
        usize::from(max_iterations),
    );

    if !success {
        return None;
    }

    let data_bytes = &output[..code.k() / 8];
    Some(unpack_bits(data_bytes, code.k()))
}

fn quantize_llr_i8(llr: &[f32]) -> Vec<i8> {
    llr.iter()
        .map(|v| {
            // labrador-ldpc treats negative LLR as hard bit 1.
            let scaled = (-v * 24.0).round();
            if scaled < -127.0 {
                -127
            } else if scaled > 127.0 {
                127
            } else if scaled == 0.0 {
                1
            } else {
                scaled as i8
            }
        })
        .collect()
}

fn select_code(fec_profile: &FecProfile) -> Option<LDPCCode> {
    match (
        usize::from(fec_profile.code_n),
        usize::from(fec_profile.code_k),
    ) {
        (128, 64) => Some(LDPCCode::TC128),
        (256, 128) => Some(LDPCCode::TC256),
        (512, 256) => Some(LDPCCode::TC512),
        (1280, 1024) => Some(LDPCCode::TM1280),
        (1536, 1024) => Some(LDPCCode::TM1536),
        (2048, 1024) => Some(LDPCCode::TM2048),
        (5120, 4096) => Some(LDPCCode::TM5120),
        (6144, 4096) => Some(LDPCCode::TM6144),
        (8192, 4096) => Some(LDPCCode::TM8192),
        _ => None,
    }
}

#[must_use]
pub fn interleave_block(input: &[u8], depth: usize) -> Vec<u8> {
    if depth <= 1 || input.is_empty() {
        return input.to_vec();
    }
    let rows = depth;
    let cols = input.len().div_ceil(rows);
    let mut out = vec![0u8; input.len()];
    let mut idx = 0usize;

    for c in 0..cols {
        for r in 0..rows {
            let src = r * cols + c;
            if src < input.len() {
                out[idx] = input[src];
                idx += 1;
            }
        }
    }
    out
}

#[must_use]
pub fn deinterleave_block(input: &[f32], depth: usize) -> Vec<f32> {
    if depth <= 1 || input.is_empty() {
        return input.to_vec();
    }
    let rows = depth;
    let cols = input.len().div_ceil(rows);
    let mut out = vec![0f32; input.len()];
    let mut idx = 0usize;

    for c in 0..cols {
        for r in 0..rows {
            let dst = r * cols + c;
            if dst < input.len() {
                out[dst] = input[idx];
                idx += 1;
            }
        }
    }
    out
}

fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &bit) in bits.iter().enumerate() {
        if bit & 1 == 1 {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    out
}

fn unpack_bits(bytes: &[u8], bit_len: usize) -> Vec<u8> {
    let mut out = vec![0u8; bit_len];
    for i in 0..bit_len {
        let b = bytes[i / 8];
        out[i] = (b >> (7 - (i % 8))) & 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldpc_round_trip_no_noise() {
        let profile = FecProfile::default();
        let bits: Vec<u8> = (0..usize::from(profile.code_k))
            .map(|i| u8::from((i % 3) == 0 || (i % 5) == 0))
            .collect();

        let enc = encode_ldpc(&bits, &profile);
        assert_eq!(enc.len(), usize::from(profile.code_n));

        let llr: Vec<f32> = enc
            .iter()
            .map(|&b| if b == 1 { 4.0 } else { -4.0 })
            .collect();

        let dec = decode_ldpc(&llr, &profile).expect("must decode");
        assert_eq!(dec, bits);
    }

    #[test]
    fn interleave_round_trip() {
        let src = [1u8, 2, 3, 4, 5, 6, 7];
        let i = interleave_block(&src, 3);
        let f: Vec<f32> = i.iter().map(|x| f32::from(*x)).collect();
        let d = deinterleave_block(&f, 3);
        let out: Vec<u8> = d.into_iter().map(|x| x as u8).collect();
        assert_eq!(out, src);
    }
}
