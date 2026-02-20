use clap::Parser;
use o4fm_audio::list_device_infos;

mod cli;
mod commands;
mod o4fm_payload;
mod wav_io;

use cli::{Cli, Commands};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Devices => print_devices()?,
        Commands::Tui(cmd) => commands::tui::run_tui_mode(&cmd)?,
        Commands::Demo(cmd) => commands::demo::run_demo(&cmd)?,
        Commands::Cpal(cmd) => commands::cpal::run_cpal_mode(&cmd)?,
        Commands::Wav(cmd) => commands::wav::run_wav_mode(&cmd)?,
        Commands::BinToWav(cmd) => commands::bin::run_bin_to_wav_mode(&cmd)?,
        Commands::WavToBin(cmd) => commands::bin::run_wav_to_bin_mode(&cmd)?,
        Commands::VoiceTx(cmd) => commands::voice::run_voice_tx_mode(&cmd)?,
        Commands::VoiceRx(cmd) => commands::voice::run_voice_rx_mode(&cmd)?,
    }

    Ok(())
}

fn print_devices() -> Result<(), Box<dyn std::error::Error>> {
    let devices = list_device_infos()?;
    println!("available audio devices:");
    for dev in devices {
        println!(
            "  id={} | in={} out={} | {}",
            dev.id, dev.supports_input, dev.supports_output, dev.name
        );
    }
    Ok(())
}
