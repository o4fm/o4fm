use o4fm_core::{LOGICAL_MAGIC, LOGICAL_MTU_BYTES, LogicalFrame};

pub(crate) fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0_u8; bits.len().div_ceil(8)];
    for (idx, &bit) in bits.iter().enumerate() {
        if bit & 1 == 1 {
            out[idx / 8] |= 1 << (7 - idx % 8);
        }
    }
    out
}

pub(crate) fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &byte in bytes {
        for bit in (0..8).rev() {
            out.push((byte >> bit) & 1);
        }
    }
    out
}

pub(super) fn extract_logical_payloads(stream: &[u8], out: &mut Vec<u8>) -> (usize, usize, usize) {
    let mut pos = 0usize;
    let mut logical_frames = 0usize;
    let mut parse_errors = 0usize;

    while pos + 8 <= stream.len() {
        if stream[pos..].len() < 4 || stream[pos..pos + 4] != LOGICAL_MAGIC {
            pos += 1;
            continue;
        }
        if pos + 8 > stream.len() {
            break;
        }
        let total_size = usize::from(u16::from_be_bytes([stream[pos + 6], stream[pos + 7]]));
        if !(1..=LOGICAL_MTU_BYTES).contains(&total_size) {
            parse_errors += 1;
            pos += 1;
            continue;
        }
        if pos + total_size > stream.len() {
            break;
        }
        let window = &stream[pos..pos + total_size];
        match LogicalFrame::decode(window) {
            Ok(logical) => {
                out.extend_from_slice(logical.payload.as_slice());
                logical_frames += 1;
                pos += total_size;
            }
            Err(_) => {
                parse_errors += 1;
                pos += 1;
            }
        }
    }

    (
        logical_frames,
        parse_errors,
        stream.len().saturating_sub(pos),
    )
}
