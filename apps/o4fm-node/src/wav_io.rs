use std::fs;
use std::path::Path;

pub(crate) fn read_wav_mono_i16(path: &str) -> Result<(Vec<i16>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("only mono WAV is currently supported".into());
    }

    let samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect()
            } else {
                let scale = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| {
                        let v = s.unwrap_or(0) as f32 / scale;
                        (v * f32::from(i16::MAX))
                            .round()
                            .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
                            as i16
                    })
                    .collect()
            }
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| {
                (s.unwrap_or(0.0) * f32::from(i16::MAX))
                    .round()
                    .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
            })
            .collect(),
    };
    Ok((samples, spec.sample_rate))
}

pub(crate) fn write_wav_mono_i16(
    out_wav: &str,
    sample_rate_hz: u32,
    samples: &[i16],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(out_wav).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_wav, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

pub(crate) fn resample_linear_i16(input: &[i16], from_hz: u32, to_hz: u32) -> Vec<i16> {
    if input.is_empty() || from_hz == to_hz || from_hz == 0 || to_hz == 0 {
        return input.to_vec();
    }
    let out_len = input.len().saturating_mul(to_hz as usize) / from_hz as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::<i16>::with_capacity(out_len);
    for i in 0..out_len {
        let pos_num = i.saturating_mul(from_hz as usize);
        let idx = pos_num / to_hz as usize;
        let frac_num = pos_num % to_hz as usize;
        let a = *input.get(idx).unwrap_or(&0);
        let b = *input.get(idx.saturating_add(1)).unwrap_or(&a);
        let frac = frac_num as f32 / to_hz as f32;
        let y = f32::from(a) * (1.0 - frac) + f32::from(b) * frac;
        out.push(y.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
    }
    out
}
