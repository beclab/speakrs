use std::path::{Path, PathBuf};

use ndarray::Array2;
use ort::session::Session;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{CachedInputShape, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError, ensure_ort_ready, with_execution_mode};

#[path = "segmentation_openvino.rs"]
mod openvino;
pub use openvino::{batched_segmentation_file_name, batched_segmentation_file_name_for};
use openvino::{primary_batched_path, tolerates_unbuildable_batched};

#[cfg(feature = "coreml")]
mod native;
#[cfg(test)]
#[path = "segmentation_openvino_tests.rs"]
mod openvino_tests;
#[cfg(feature = "coreml")]
mod parallel;
mod run;
mod tensor;

/// Errors that can occur during segmentation inference
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SegmentationError {
    /// ONNX Runtime error
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Streaming channel was closed before all windows were sent
    #[error("receiver disconnected")]
    Disconnected(#[from] crossbeam_channel::SendError<Array2<f32>>),
    /// Internal segmentation invariant was violated
    #[error("{context}: {message}")]
    Invariant {
        /// Which step failed
        context: &'static str,
        /// Invariant failure details
        message: String,
    },
    /// Model output was missing or had an unexpected shape
    #[error("{context}: {message}")]
    MalformedOutput {
        /// Which output extraction step failed
        context: &'static str,
        /// Output validation details
        message: String,
    },
    /// Background worker panicked
    #[error("{worker} thread panicked")]
    WorkerPanic {
        /// Worker or thread name
        worker: String,
    },
}

// seg models exported with EnumeratedShapes for batch 1-32 and b64
const PRIMARY_BATCH_SIZE: usize = 32;
#[cfg(feature = "coreml")]
const LARGE_BATCH_SIZE: usize = 64;

/// Sliding-window segmentation model (pyannote segmentation-3.0)
pub struct SegmentationModel {
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    #[cfg(feature = "coreml")]
    native_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_large_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    cached_single_input_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_batch_input_shape: CachedInputShape,
    input_buffer: ndarray::Array3<f32>,
    primary_batch_input_buffer: ndarray::Array3<f32>,
    window_samples: usize,
    step_samples: usize,
    sample_rate: usize,
}

// SAFETY: SegmentationModel is only used from one thread at a time via &mut self
// SAFETY: the non-Send fields contain Objective-C objects that are only moved, not shared
// SAFETY: SharedCoreMlModel is already Send + Sync
#[cfg(feature = "coreml")]
unsafe impl Send for SegmentationModel {}

impl SegmentationModel {
    /// Whether segmentation runs batched, which is the session and not the file.
    ///
    /// A consumer that reports this had to ask whether the prepared model exists, and that
    /// answer is now one step short of the truth: the file can be there and its session can
    /// have been declined, which is exactly the case the load path started tolerating.
    pub fn is_batched(&self) -> bool {
        self.primary_batched_session.is_some()
    }

    /// Load a segmentation-3.0 ONNX model
    pub fn new(model_path: impl AsRef<Path>, step_duration: f32) -> Result<Self, ModelLoadError> {
        Self::with_mode(model_path, step_duration, ExecutionMode::Cpu)
    }

    /// Load a segmentation-3.0 ONNX model with the requested execution mode
    pub fn with_mode(
        model_path: impl AsRef<Path>,
        step_duration: f32,
        mode: ExecutionMode,
    ) -> Result<Self, ModelLoadError> {
        mode.validate()?;
        ensure_ort_ready()?;

        let model_path = model_path.as_ref();
        let sample_rate = 16000;
        let window_duration = 10.0;
        let window_samples = (window_duration * sample_rate as f32) as usize;
        let step_samples = (step_duration * sample_rate as f32) as usize;

        #[cfg(feature = "coreml")]
        if matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            Self::validate_native_coreml_assets(model_path, mode)?;
        }

        macro_rules! timed {
            ($expr:expr) => {{
                let start = std::time::Instant::now();
                let value = $expr;
                (value, start.elapsed())
            }};
        }

        let (session, session_elapsed) = timed!(Self::build_session(model_path, mode)?);
        let (primary_batched_session, primary_batched_elapsed) =
            timed!(match primary_batched_path(model_path, mode) {
                None => None,
                // On OpenVINO a batched model that will not build turns batching off; it
                // does not stop the pipeline loading. Absence was already the safe direction
                // to fall, and this is the same outcome discovered one step later. There the
                // file is derived at startup by the consumer from whatever export is in the
                // cache, so a new export shape produces a model that passes onnx's checker
                // and that the GPU plugin then refuses; propagating that takes down every
                // installation that derives it, the next time the weights change, for a
                // feature whose absence costs speed and nothing else.
                //
                // Every other backend keeps the behaviour it had before this one was added:
                // error propagates and the load fails. Their batched export ships with the
                // weights, so one that is present and will not build is a damaged download,
                // and stopping on it says so where falling back would not. An earlier version
                // of this fell back on every mode -- a corrupt export on CUDA stopped failing
                // the load and started costing throughput in silence, which is a change to
                // backends that adding this one must leave alone.
                //
                // The same asymmetry, for the same reason, is why the embedding loader does
                // not fall back at all: nothing derives its exports either.
                Some(path) => Self::build_session(&path, mode)
                    .map(Some)
                    .or_else(|error| {
                        if tolerates_unbuildable_batched(mode) {
                            tracing::warn!(
                                model = %path.display(),
                                %error,
                                "the batched segmentation model would not build; \
                                 running segmentation one window at a time"
                            );
                            Ok(None)
                        } else {
                            Err(error)
                        }
                    })?,
            });
        #[cfg(feature = "coreml")]
        let (native_session, native_session_elapsed) =
            timed!(Self::load_native_coreml(model_path, mode)?);
        #[cfg(feature = "coreml")]
        let (native_batched_session, native_batched_elapsed) =
            timed!(Self::load_native_coreml_batched(model_path, mode)?);
        #[cfg(feature = "coreml")]
        let (native_large_batched_session, native_large_batched_elapsed) =
            timed!(Self::load_native_coreml_large_batched(model_path, mode)?);

        #[cfg(feature = "coreml")]
        if matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            if native_session.is_none() {
                return Err(ModelLoadError::MissingNativeAsset {
                    mode,
                    path: Self::resolve_coreml_path(model_path, mode)
                        .unwrap_or_else(|| model_path.to_path_buf()),
                });
            }
            if native_batched_session.is_none() {
                return Err(ModelLoadError::MissingNativeAsset {
                    mode,
                    path: Self::resolve_batched_coreml_path(model_path, mode, PRIMARY_BATCH_SIZE)
                        .unwrap_or_else(|| model_path.to_path_buf()),
                });
            }
            if native_large_batched_session.is_none() {
                return Err(ModelLoadError::MissingNativeAsset {
                    mode,
                    path: Self::resolve_batched_coreml_path(model_path, mode, LARGE_BATCH_SIZE)
                        .unwrap_or_else(|| model_path.to_path_buf()),
                });
            }
        }

        #[cfg(feature = "coreml")]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + native_session_elapsed
                + native_batched_elapsed
                + native_large_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_batched_ms = primary_batched_elapsed.as_millis(),
                native_single_ms = native_session_elapsed.as_millis(),
                native_b32_ms = native_batched_elapsed.as_millis(),
                native_b64_ms = native_large_batched_elapsed.as_millis(),
                total_ms,
                "Segmentation model init",
            );
        }
        #[cfg(not(feature = "coreml"))]
        {
            let total_ms = (session_elapsed + primary_batched_elapsed).as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_batched_ms = primary_batched_elapsed.as_millis(),
                total_ms,
                "Segmentation model init",
            );
        }

        Ok(Self {
            mode,
            session,
            primary_batched_session,
            #[cfg(feature = "coreml")]
            native_session,
            #[cfg(feature = "coreml")]
            native_batched_session,
            #[cfg(feature = "coreml")]
            native_large_batched_session,
            #[cfg(feature = "coreml")]
            cached_single_input_shape: CachedInputShape::new("input", &[1, 1, window_samples]),
            #[cfg(feature = "coreml")]
            cached_batch_input_shape: CachedInputShape::new(
                "input",
                &[PRIMARY_BATCH_SIZE, 1, window_samples],
            ),
            input_buffer: ndarray::Array3::zeros((1, 1, window_samples)),
            primary_batch_input_buffer: ndarray::Array3::zeros((
                PRIMARY_BATCH_SIZE,
                1,
                window_samples,
            )),
            window_samples,
            step_samples,
            sample_rate,
        })
    }

    fn build_session(model_path: &Path, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(Self::available_threads().min(6))?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn available_threads() -> usize {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Number of audio samples per sliding window
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Number of audio samples the window advances each step
    pub fn step_samples(&self) -> usize {
        self.step_samples
    }

    /// Step size in seconds
    pub fn step_seconds(&self) -> f64 {
        self.step_samples as f64 / self.sample_rate as f64
    }

    /// Execution mode this model was loaded with
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }
}

fn batched_model_path(model_path: &Path, batch_size: usize) -> Option<PathBuf> {
    let path = model_path;
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}
