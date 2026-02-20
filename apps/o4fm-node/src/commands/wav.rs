use o4fm_audio::{AudioIo, WavFileAudio};

use crate::cli::WavArgs;

pub(crate) fn run_wav_mode(args: &WavArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input = args.r#in.as_str();
    let output = args.out.as_str();

    let sample_rate_hz = args.sample_rate;
    let frame_samples = args.frame_samples.max(1);
    let gain = args.gain;

    let mut audio = WavFileAudio::open(input, output, sample_rate_hz)?;
    let mut buf = vec![0_i16; frame_samples];
    let mut total = 0usize;

    println!(
        "o4fm-node wav mode: in={input} out={output} sample_rate={sample_rate_hz} frame={frame_samples} gain={gain:.3}"
    );

    loop {
        let n = audio.read_frame(&mut buf)?;
        if n == 0 {
            break;
        }

        if (gain - 1.0).abs() > f32::EPSILON {
            for sample in &mut buf[..n] {
                let v = f32::from(*sample) * gain;
                *sample = v.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            }
        }

        audio.write_frame(&buf[..n])?;
        total += n;
    }

    audio.finalize()?;
    println!("wav session done");
    println!("  samples_processed: {total}");

    Ok(())
}
