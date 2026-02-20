use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RadioBusy {
    #[default]
    Idle,
    Busy,
}

#[derive(Debug, Error)]
pub enum RadioIoError {
    #[error("not connected")]
    NotConnected,
    #[error("I/O failure")]
    Io,
}

pub trait RadioIo {
    /// Set PTT state.
    ///
    /// # Errors
    /// Returns an error if radio I/O backend fails.
    fn set_ptt(&mut self, enabled: bool) -> Result<(), RadioIoError>;
    /// Read current COR/busy state.
    ///
    /// # Errors
    /// Returns an error if radio I/O backend fails.
    fn read_cor(&mut self) -> Result<RadioBusy, RadioIoError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MockRadioIo {
    ptt: bool,
    cor: RadioBusy,
}

impl MockRadioIo {
    #[must_use]
    pub fn ptt(&self) -> bool {
        self.ptt
    }

    pub fn set_mock_cor(&mut self, cor: RadioBusy) {
        self.cor = cor;
    }
}

impl RadioIo for MockRadioIo {
    fn set_ptt(&mut self, enabled: bool) -> Result<(), RadioIoError> {
        self.ptt = enabled;
        Ok(())
    }

    fn read_cor(&mut self) -> Result<RadioBusy, RadioIoError> {
        Ok(self.cor)
    }
}
