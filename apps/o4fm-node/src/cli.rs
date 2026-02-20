use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "o4fm-node",
    about = "O4FM reference node",
    arg_required_else_help = true
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    Devices,
    Tui(TuiArgs),
    Demo(DemoArgs),
    Cpal(CpalArgs),
    Wav(WavArgs),
    #[command(name = "bin-to-wav")]
    BinToWav(Bin2WavArgs),
    #[command(name = "wav-to-bin")]
    WavToBin(Wav2BinArgs),
    VoiceTx(VoiceTxArgs),
    VoiceRx(VoiceRxArgs),
}

#[derive(Debug, Args)]
pub(crate) struct TuiArgs {
    #[arg(long, default_value_t = 10)]
    pub(crate) tick_hz: u64,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum Pipeline {
    Digital,
    Passthrough,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum LpMode {
    Text,
    Voice,
    Ip,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum VoiceDspArg {
    Basic,
    Passthrough,
}

#[derive(Debug, Args)]
pub(crate) struct DemoArgs {
    #[arg(long, default_value = "target/o4fm-node-demo.wav")]
    pub(crate) demo_wave_out: String,
    #[arg(long, default_value_t = 7_000)]
    pub(crate) voice_bitrate: u32,
}

#[derive(Debug, Args)]
pub(crate) struct CpalArgs {
    #[arg(long, value_enum, default_value_t = Pipeline::Digital)]
    pub(crate) pipeline: Pipeline,
    #[arg(long, default_value_t = 48_000)]
    pub(crate) sample_rate: u32,
    #[arg(long, default_value_t = 1)]
    pub(crate) channels: u16,
    #[arg(long, default_value_t = 480)]
    pub(crate) frame_samples: usize,
    #[arg(long, default_value_t = 10)]
    pub(crate) seconds: u64,
    #[arg(long, default_value_t = 7_000)]
    pub(crate) voice_bitrate: u32,
    #[arg(long)]
    pub(crate) input_device_id: Option<String>,
    #[arg(long)]
    pub(crate) output_device_id: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct WavArgs {
    #[arg(long)]
    pub(crate) r#in: String,
    #[arg(long)]
    pub(crate) out: String,
    #[arg(long, default_value_t = 48_000)]
    pub(crate) sample_rate: u32,
    #[arg(long, default_value_t = 480)]
    pub(crate) frame_samples: usize,
    #[arg(long, default_value_t = 1.0)]
    pub(crate) gain: f32,
}

#[derive(Debug, Args)]
pub(crate) struct Bin2WavArgs {
    #[arg(long)]
    pub(crate) in_bin: String,
    #[arg(long)]
    pub(crate) out_wav: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) inter_frame_silence_ms: u32,
    #[arg(long, default_value = "NOCALL")]
    pub(crate) callsign: String,
    #[arg(long, value_parser = parse_u64_auto, default_value = "0x0")]
    pub(crate) lp_flags: u64,
    #[arg(long, value_enum, default_value_t = LpMode::Text)]
    pub(crate) lp_mode: LpMode,
    #[arg(long, default_value_t = 7_000)]
    pub(crate) voice_bitrate: u32,
    #[arg(long, default_value_t = 0)]
    pub(crate) profile_id: u8,
}

#[derive(Debug, Args)]
pub(crate) struct Wav2BinArgs {
    #[arg(long)]
    pub(crate) in_wav: String,
    #[arg(long)]
    pub(crate) out_bin: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) zero_threshold: i16,
    #[arg(long, default_value_t = 7_000)]
    pub(crate) voice_bitrate: u32,
    #[arg(long)]
    pub(crate) profile_id: Option<u8>,
}

#[derive(Debug, Args)]
pub(crate) struct VoiceTxArgs {
    #[arg(long)]
    pub(crate) in_wav: String,
    #[arg(long)]
    pub(crate) out_wav: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) inter_frame_silence_ms: u32,
    #[arg(long, default_value = "NOCALL")]
    pub(crate) callsign: String,
    #[arg(long, value_parser = parse_u64_auto, default_value = "0x0")]
    pub(crate) lp_flags: u64,
    #[arg(long, default_value_t = 7000)]
    pub(crate) opus_bitrate: i32,
    #[arg(long, default_value_t = 20)]
    pub(crate) opus_frame_ms: usize,
    #[arg(long, value_enum, default_value_t = VoiceDspArg::Basic)]
    pub(crate) voice_dsp: VoiceDspArg,
}

#[derive(Debug, Args)]
pub(crate) struct VoiceRxArgs {
    #[arg(long)]
    pub(crate) in_wav: String,
    #[arg(long)]
    pub(crate) out_wav: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) zero_threshold: i16,
    #[arg(long, default_value_t = 7000)]
    pub(crate) opus_bitrate: i32,
    #[arg(long, value_enum, default_value_t = VoiceDspArg::Basic)]
    pub(crate) voice_dsp: VoiceDspArg,
}

fn parse_u64_auto(v: &str) -> Result<u64, String> {
    let trimmed = v.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex value `{v}`: {e}"))
    } else {
        trimmed
            .parse::<u64>()
            .map_err(|e| format!("invalid integer value `{v}`: {e}"))
    }
}
