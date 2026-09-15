pub(crate) mod embedding;
pub(crate) mod segmentation;

/// The file name OpenVINO looks for when it batches segmentation, re-exported because the
/// module itself is crate-private and the name has to be reachable by whoever writes the file.
pub use segmentation::{batched_segmentation_file_name, batched_segmentation_file_name_for};

use std::collections::HashSet;
use std::sync::Mutex;

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::ffi::CStr;
use std::fmt;
#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;

pub use embedding::EmbeddingModel;
pub use segmentation::{SegmentationError, SegmentationModel};

#[cfg(feature = "coreml")]
pub(crate) mod coreml;

use ort::ep;
use ort::session::builder::SessionBuilder;

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
static ORT_RUNTIME_INIT: OnceLock<Result<(), OrtRuntimeError>> = OnceLock::new();

/// CoreML compute unit selection for chunk embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMlComputeUnits {
    /// Use all available compute units: CPU + GPU + Neural Engine (default)
    #[default]
    All,
    /// Use CPU + Neural Engine only (skip GPU)
    CpuAndNeuralEngine,
}

#[cfg(feature = "coreml")]
impl CoreMlComputeUnits {
    pub(crate) fn to_ml_compute_units(self) -> objc2_core_ml::MLComputeUnits {
        match self {
            Self::All => crate::inference::coreml::CoreMlModel::default_compute_units(),
            Self::CpuAndNeuralEngine => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
        }
    }
}

/// Which backend and acceleration to use for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// CPU-only via ORT (portable, slowest)
    Cpu,
    /// Native CoreML with FP32 precision and ~1s step
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    CoreMl,
    /// Native CoreML with W8A16 segmentation and ~2s step
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    CoreMlFast,
    /// NVIDIA GPU with concurrent fused seg+emb via crossbeam
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    Cuda,
    /// NVIDIA GPU with concurrent fused seg+emb and ~2s step
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    CudaFast,
    /// AMD GPU via ONNX Runtime's MIGraphX execution provider
    #[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
    MiGraphX,
    /// Intel CPU, integrated GPU, discrete GPU or NPU via ONNX Runtime's OpenVINO
    /// execution provider
    ///
    /// `device_type` reaches OpenVINO verbatim: `CPU`, `GPU`, `GPU.0`, `GPU.1`, `NPU`, or a
    /// heterogeneous combination such as `HETERO:NPU,GPU`. It is a field rather than one
    /// variant per device because a machine carrying both an integrated and a discrete Intel
    /// GPU does not guarantee which of them is `GPU.0` -- only the caller, on the machine, can
    /// resolve that.
    #[cfg_attr(docsrs, doc(cfg(feature = "openvino")))]
    OpenVino {
        /// OpenVINO device string, passed through unchanged
        device_type: &'static str,
    },
}

impl ExecutionMode {
    /// Returns true when this mode uses native CoreML execution
    pub const fn is_coreml(self) -> bool {
        matches!(self, Self::CoreMl | Self::CoreMlFast)
    }

    /// Returns true when this mode uses CUDA execution
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda | Self::CudaFast)
    }

    /// Returns true when this mode uses the MIGraphX execution provider
    pub const fn is_migraphx(self) -> bool {
        matches!(self, Self::MiGraphX)
    }

    /// Returns true when this mode uses the OpenVINO execution provider
    pub const fn is_openvino(self) -> bool {
        matches!(self, Self::OpenVino { .. })
    }

    pub(crate) fn validate(self) -> Result<(), ExecutionModeError> {
        if self == Self::Cpu {
            return Ok(());
        }

        if self.is_coreml() {
            #[cfg(feature = "coreml")]
            {
                return Ok(());
            }

            #[cfg(not(feature = "coreml"))]
            {
                return Err(ExecutionModeError {
                    mode: self,
                    feature: "coreml",
                });
            }
        }

        if self.is_migraphx() {
            #[cfg(feature = "migraphx")]
            {
                return Ok(());
            }

            #[cfg(not(feature = "migraphx"))]
            {
                return Err(ExecutionModeError {
                    mode: self,
                    feature: "migraphx",
                });
            }
        }

        if let Self::OpenVino { device_type } = self {
            // 🔴 Only the empty string. ONNX Runtime does not check this either -- ort passes
            // device_type straight into the provider options -- and the set of legal names
            // lives in ONNX Runtime's C++ side and in OpenVINO, where it grows: AUTO, MULTI,
            // HETERO and BATCH prefixes, GPU.N positions, a batch size in brackets. Refusing
            // what this crate does not recognise would refuse a syntax that arrives later,
            // and it would refuse it here, where nobody can work around it.
            //
            // Empty is different: it names no device at all, and the provider's answer to it
            // is an error a long way from the call that caused it.
            if device_type.trim().is_empty() {
                return Err(ExecutionModeError {
                    mode: self,
                    feature: "openvino",
                });
            }
        }

        if self.is_openvino() {
            #[cfg(feature = "openvino")]
            {
                return Ok(());
            }

            #[cfg(not(feature = "openvino"))]
            {
                return Err(ExecutionModeError {
                    mode: self,
                    feature: "openvino",
                });
            }
        }

        debug_assert!(self.is_cuda(), "unsupported execution mode: {self:?}");

        #[cfg(feature = "cuda")]
        {
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(ExecutionModeError {
                mode: self,
                feature: "cuda",
            })
        }
    }

    /// An OpenVINO mode for a device this process discovered at runtime.
    ///
    /// 🔴 The device a machine has is not known until the process is on it -- `GPU.0` and
    /// `GPU.1` are positions, and which one is the discrete card is the machine's business --
    /// so the string usually arrives from argv, an environment variable, or a probe. The
    /// variant holds a `&'static str` because `ExecutionMode` is `Copy` and is passed by value
    /// in dozens of places, which leaves a caller with a runtime string no way in except to
    /// leak one. Every caller then writes that line itself, and writes it differently.
    ///
    /// ⚠️ This leaks too, once per distinct device string, and never frees. A process uses one
    /// or two, so the total is bounded by how many different devices it is asked for rather
    /// than by how many times it asks. Repeated calls with the same string reuse the first.
    pub fn openvino(device: &str) -> Self {
        static SEEN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let device_type = match seen.get(device) {
            Some(existing) => existing,
            None => {
                let leaked: &'static str = Box::leak(device.to_owned().into_boxed_str());
                seen.insert(leaked);
                leaked
            }
        };
        Self::OpenVino { device_type }
    }

    /// This mode named for a human, with the device when there is one: `openvino:GPU.1`.
    ///
    /// 🔴 Separate from `as_str` rather than replacing it. `as_str` is the backend's name and
    /// is `const`, so it cannot carry a device in the first place, and consumers already branch
    /// on it and key caches by it. This is the string a log line wants, and the one consumer
    /// that needed it was building it by hand -- and leaking it -- beside a mode it had just
    /// built.
    pub fn label(self) -> String {
        match self {
            Self::OpenVino { device_type } => format!("openvino:{device_type}"),
            other => other.as_str().to_owned(),
        }
    }

    /// Lowercase identifier used in logs, docs, and user-facing errors
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CoreMl => "coreml",
            Self::CoreMlFast => "coreml-fast",
            Self::Cuda => "cuda",
            Self::CudaFast => "cuda-fast",
            Self::MiGraphX => "migraphx",
            Self::OpenVino { .. } => "openvino",
        }
    }
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What an OpenVINO device string resolves to, as far as decisions in this crate are concerned.
///
/// 🔴 The split is by decision, not by kind of hardware. `HETERO:NPU,GPU` and `MULTI:CPU,NPU`
/// are both "several devices", and they want opposite answers: the first reaches the GPU
/// plugin and the second cannot. What every caller here actually asks is whether the GPU
/// plugin is in play.
///
/// ⚠️ It cannot tell a discrete card from an integrated one. `GPU`, `GPU.0` and `GPU.1` are
/// positions in a list, not kinds of hardware, and only a caller standing on the machine knows
/// which is which. Anything that needs the distinction -- the precision the embedding models
/// are given, for one -- is answered for every GPU or for none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OvTarget {
    /// `GPU`, `GPU.1`
    GpuOnly,
    /// A device list that may reach the GPU plugin: `HETERO:NPU,GPU`, `AUTO:GPU,CPU`,
    /// `BATCH:GPU(4)`
    ///
    /// Also where anything unreadable lands -- an unknown prefix, a name this crate cannot
    /// parse -- because that is the cheaper way to be wrong. So this says "may reach the GPU
    /// plugin", not "names a GPU": `NonGpuComposite` is the one that makes a claim.
    GpuComposite,
    /// A device list whose every name was read, and none of them a GPU: `MULTI:CPU,NPU`
    ///
    /// 🔴 Every name read, not merely no GPU found. A list holding a name this crate cannot
    /// read is not this: see `ov_target` for why the difference decides whether a machine
    /// loads at all.
    NonGpuComposite,
    /// `CPU`
    Cpu,
    /// `NPU`
    Npu,
    /// Bare `AUTO` or bare `MULTI`: a decision OpenVINO makes at runtime, on a machine this
    /// crate cannot see. Kept as its own value rather than folded into either side, because
    /// what to do about it is a judgement written down at each call site, not a fact.
    UnresolvedAuto,
}

/// The device names this crate recognises inside a device string.
pub(crate) fn ov_device_token(token: &str) -> Option<&'static str> {
    // 🔴 The batch size comes off first. `BATCH:GPU(4)` sets the batch explicitly -- OpenVINO's
    // own documentation gives `BATCH:GPU(16)` and `BATCH:CPU(16)` as examples -- so a token
    // compared whole reads `GPU(4)` as some device that is not a GPU, and the one form whose
    // whole purpose is batching would be the one that loses it.
    let token = token.trim();
    let token = match token.split_once('(') {
        Some((head, tail)) if tail.ends_with(')') && !tail.is_empty() => head.trim_end(),
        _ => token,
    };
    let base = token.split_once('.').map_or(token, |(head, _)| head);
    match base {
        "GPU" => Some("GPU"),
        "CPU" => Some("CPU"),
        "NPU" => Some("NPU"),
        _ => None,
    }
}

/// Resolve a device string once, so that every decision below reads the same answer.
///
/// The one place `AUTO` / `MULTI` / `HETERO` / `BATCH` / `GPU.1` / a batch size in brackets are
/// interpreted. A device syntax that arrives later is a change here and nowhere else.
pub(crate) fn ov_target(device_type: &str) -> OvTarget {
    let device_type = device_type.trim();
    match device_type.split_once(':') {
        Some((prefix, list)) => {
            let prefix = prefix.trim();
            if !matches!(prefix, "AUTO" | "MULTI" | "HETERO" | "BATCH") {
                // Not a form this crate knows. Treated as reaching the plugin: that direction
                // costs batching when the substitute is missing, the other costs a session
                // that will not build.
                return OvTarget::GpuComposite;
            }
            if list.trim().is_empty() {
                OvTarget::UnresolvedAuto
            } else if list.split(',').any(|t| ov_device_token(t) == Some("GPU")) {
                OvTarget::GpuComposite
            } else if list.split(',').all(|t| ov_device_token(t).is_some()) {
                OvTarget::NonGpuComposite
            } else {
                // 🔴 `NonGpuComposite` is claimed only when every name in the list was read.
                // Reaching it by finding no GPU folds together two lists that look identical
                // from here: one that names no GPU, and one carrying a name this crate cannot
                // read. `AUTO:-CPU` is the second -- OpenVINO's documented way to say "choose
                // automatically, minus the CPU", which on an Intel machine is how you ask for
                // the card -- and `NonGpuComposite` is the one answer that both takes the
                // stock export and declines to survive its refusal, so the string that asks
                // for the GPU most plainly would be the one that fails to load.
                //
                // Same direction as the unknown prefix above, for the same reason: unreadable
                // costs batching when the substitute is missing, and the load when it is not.
                OvTarget::GpuComposite
            }
        }
        None => match ov_device_token(device_type) {
            Some("GPU") => OvTarget::GpuOnly,
            Some("CPU") => OvTarget::Cpu,
            Some("NPU") => OvTarget::Npu,
            _ if matches!(device_type, "AUTO" | "MULTI") => OvTarget::UnresolvedAuto,
            // Same reasoning as the unknown prefix above.
            _ => OvTarget::GpuComposite,
        },
    }
}

/// Whether this mode certainly reaches OpenVINO's GPU plugin.
///
/// 🔴 Bare `AUTO` is deliberately not here, and `may_reach_openvino_gpu` is the one that says
/// it might be. Which way to guess about `AUTO` depends on what the answer is used for, and the
/// two callers here want opposite guesses:
///
/// - **Which model file to load** uses this one. Guessing "GPU" sends a machine with no GPU to
///   a derived model nobody provisioned, costing it the batching it would have had at 62.1 ms
///   a window. Guessing "not GPU" hands a GPU the stock export, which it refuses -- and that
///   refusal is survivable, because `tolerates_unbuildable_batched` includes `AUTO`.
/// - **Whether the embedding models get FP32** uses the other one. Guessing "not GPU" sends a
///   discrete card back to the default precision, where these models return
///   `CL_OUT_OF_RESOURCES` out of `clFinish` -- the failure FP32 exists to answer. Guessing
///   "GPU" costs a processor a precision it did not need.
///
/// One costs speed, the other costs the load. They are not the same question and cannot share
/// an answer; an earlier version of this had them share this predicate, which quietly took FP32
/// away from bare `AUTO`.
pub(crate) fn openvino_gpu_plugin(mode: ExecutionMode) -> bool {
    match mode {
        ExecutionMode::OpenVino { device_type } => {
            matches!(
                ov_target(device_type),
                OvTarget::GpuOnly | OvTarget::GpuComposite
            )
        }
        _ => false,
    }
}

/// Whether this mode may reach OpenVINO's GPU plugin, counting the case nobody can resolve.
///
/// `openvino_gpu_plugin` plus bare `AUTO`. For decisions whose wrong answer costs a session
/// rather than its speed -- see the note there for why the two directions differ.
pub(crate) fn may_reach_openvino_gpu(mode: ExecutionMode) -> bool {
    match mode {
        ExecutionMode::OpenVino { device_type } => matches!(
            ov_target(device_type),
            OvTarget::GpuOnly | OvTarget::GpuComposite | OvTarget::UnresolvedAuto
        ),
        _ => false,
    }
}

/// Errors that can occur while loading a model or initializing ONNX Runtime
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelLoadError {
    /// Requested execution mode is not supported by this build
    #[error(transparent)]
    UnsupportedExecutionMode(#[from] ExecutionModeError),
    /// ONNX Runtime could not be prepared for this process
    #[error(transparent)]
    Runtime(#[from] OrtRuntimeError),
    /// ONNX Runtime returned an error after initialization completed
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// A required native model asset is missing for the selected execution mode
    #[error("{mode} requires native asset `{path}`")]
    MissingNativeAsset {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The missing compiled CoreML bundle path
        path: PathBuf,
    },
    /// A required native model asset exists but failed to load
    #[error("{mode} failed to load native asset `{path}`: {message}")]
    NativeAssetLoad {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The compiled CoreML bundle path that failed to load
        path: PathBuf,
        /// The backend load error
        message: String,
    },
}

/// Errors that can occur while preparing the process-wide ONNX Runtime environment
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum OrtRuntimeError {
    /// Dynamic runtime discovery or validation failed before `ort` could initialize
    #[error(transparent)]
    Dynamic(#[from] DynamicRuntimeError),
    /// `ort::init_from` failed after runtime validation succeeded
    #[error("failed to initialize ONNX Runtime: {message}")]
    Initialization {
        /// The initialization error returned by `ort`
        message: String,
    },
}

/// Errors from locating or validating the dynamic ONNX Runtime library
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DynamicRuntimeError {
    /// No candidate runtime library was found
    #[error(
        "missing ONNX Runtime dynamic library `{library_name}`; set `ORT_DYLIB_PATH` or place it next to the test/binary\nsearched: {searched}"
    )]
    Missing {
        /// The platform-specific dynamic library filename
        library_name: &'static str,
        /// The candidate paths checked before giving up
        searched: String,
    },
    /// Loading the requested runtime library failed
    #[error("failed to load ONNX Runtime dynamic library at `{path}`: {message}")]
    Load {
        /// The path that failed to load
        path: PathBuf,
        /// The dynamic loader error
        message: String,
    },
    /// The requested runtime library does not export `OrtGetApiBase`
    #[error("ONNX Runtime dynamic library at `{path}` does not export `OrtGetApiBase`")]
    MissingApiBase {
        /// The path that was missing the required symbol
        path: PathBuf,
    },
    /// The requested runtime library returned a null API base pointer
    #[error("ONNX Runtime dynamic library at `{path}` returned a null `OrtApiBase`")]
    NullApiBase {
        /// The path that returned a null API pointer
        path: PathBuf,
    },
    /// The requested runtime library is older than the `ort` crate expects
    #[error(
        "ONNX Runtime dynamic library at `{path}` is too old; expected >= 1.{required_minor}.x, got `{found_version}`"
    )]
    IncompatibleVersion {
        /// The incompatible runtime library path
        path: PathBuf,
        /// The minimum ONNX Runtime minor version required by `ort`
        required_minor: u32,
        /// The version reported by the discovered runtime library
        found_version: String,
    },
}

/// Errors from requesting an execution mode that is not supported in the current build
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
#[error("{mode} requires the `{feature}` Cargo feature")]
pub struct ExecutionModeError {
    mode: ExecutionMode,
    feature: &'static str,
}

impl From<ExecutionModeError> for ort::Error {
    fn from(error: ExecutionModeError) -> Self {
        ort::Error::new(error.to_string())
    }
}

/// Map an execution mode to ORT execution providers
///
/// CoreML modes use ORT CPU for any sessions that still go through ORT such as FBANK,
/// While segmentation and embedding tail sessions use native CoreML directly
pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ort::Error> {
    with_execution_mode_precision(builder, mode, None)
}

/// As `with_execution_mode`, with an OpenVINO inference precision for this one session.
///
/// Per session, not per pipeline, because no single precision works for the whole of it.
/// Measured on Arc Pro B70 (Battlemage, driver 26.22.38646.4, OpenVINO 2025.4.1): the
/// embedding models return CL_OUT_OF_RESOURCES from clFinish at the default precision and
/// run correctly at FP32, while segmentation is the other way round -- FP32 takes it from
/// 6.0 s to 6.2 s per window and stops the batched graph compiling at all.
///
/// Per session but NOT per device, and that is a limit rather than a decision: only the
/// discrete card was measured to need this, and nothing here can tell a discrete card from
/// an integrated one -- `GPU`, `GPU.0` and `GPU.1` are positions, not kinds. So the caller
/// that knows would have to say, and no caller does. What that costs on the integrated part
/// is unmeasured; the figures published for it were taken with FP32 already in force.
///
/// `precision` is ignored by every mode but OpenVINO, which is why it is a parameter here
/// rather than a field on the mode: it describes how one session is built, not what the
/// pipeline was asked to run on.
pub fn with_execution_mode_precision(
    builder: SessionBuilder,
    mode: ExecutionMode,
    precision: Option<&str>,
) -> Result<SessionBuilder, ort::Error> {
    mode.validate()?;

    match mode {
        ExecutionMode::Cpu | ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => Ok(builder
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?),
        ExecutionMode::Cuda | ExecutionMode::CudaFast => {
            #[cfg(feature = "cuda")]
            {
                Ok(builder.with_execution_providers([ep::CUDA::default()
                    .with_device_id(0)
                    .with_tf32(true)
                    .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
                    .with_conv_max_workspace(true)
                    .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                    .with_prefer_nhwc(true)
                    .build()
                    .error_on_failure()])?)
            }

            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("mode validation rejects CUDA modes without the `cuda` feature")
            }
        }
        ExecutionMode::MiGraphX => {
            #[cfg(feature = "migraphx")]
            {
                Ok(builder.with_execution_providers([ep::MIGraphX::default()
                    .with_device_id(0)
                    .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                    .build()
                    .error_on_failure()])?)
            }

            #[cfg(not(feature = "migraphx"))]
            {
                unreachable!("mode validation rejects MIGraphX mode without the `migraphx` feature")
            }
        }
        ExecutionMode::OpenVino { device_type } => {
            #[cfg(feature = "openvino")]
            {
                let provider = ep::OpenVINO::default().with_device_type(device_type);
                let provider = match precision {
                    Some(precision) => provider.with_precision(precision),
                    None => provider,
                };
                Ok(builder.with_execution_providers([provider.build().error_on_failure()])?)
            }

            #[cfg(not(feature = "openvino"))]
            {
                let _ = (device_type, precision);
                unreachable!("mode validation rejects OpenVINO mode without the `openvino` feature")
            }
        }
    }
}

pub(crate) fn ensure_ort_ready() -> Result<(), ModelLoadError> {
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    {
        let init_result = ORT_RUNTIME_INIT.get_or_init(|| OrtRuntimeLoader::new().initialize());
        init_result.clone()?;
    }

    Ok(())
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
struct OrtRuntimeLoader {
    library_name: &'static str,
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
impl OrtRuntimeLoader {
    fn new() -> Self {
        Self {
            library_name: Self::default_library_name(),
        }
    }

    fn initialize(&self) -> Result<(), OrtRuntimeError> {
        let path = self.resolve_library_path()?;
        self.validate_library(&path)?;

        ort::init_from(&path)
            .map(|builder| {
                builder.commit();
            })
            .map_err(|error| OrtRuntimeError::Initialization {
                message: error.to_string(),
            })
    }

    fn resolve_library_path(&self) -> Result<PathBuf, DynamicRuntimeError> {
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH")
            && !path.is_empty()
        {
            let path = PathBuf::from(path);
            return path.exists().then_some(path.clone()).ok_or_else(|| {
                DynamicRuntimeError::Missing {
                    library_name: self.library_name,
                    searched: path.display().to_string(),
                }
            });
        }

        let candidates = self.candidate_paths();
        candidates
            .iter()
            .find(|path| path.exists())
            .cloned()
            .ok_or_else(|| DynamicRuntimeError::Missing {
                library_name: self.library_name,
                searched: Self::format_paths(&candidates),
            })
    }

    fn candidate_paths(&self) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        if let Ok(exe) = std::env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            candidates.push(exe_dir.join(self.library_name));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join(self.library_name));
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join(self.library_name));
            candidates.push(cwd.join("target/debug").join(self.library_name));
            candidates.push(cwd.join("target/debug/deps").join(self.library_name));
            candidates.push(cwd.join("target/release").join(self.library_name));
            candidates.push(cwd.join("target/release/deps").join(self.library_name));
        }

        dedup_paths(candidates)
    }

    fn validate_library(&self, path: &Path) -> Result<(), DynamicRuntimeError> {
        // safety: we only open the candidate runtime long enough to validate its exported API
        let library = unsafe { libloading::Library::new(path) }.map_err(|error| {
            DynamicRuntimeError::Load {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;

        // safety: the library handle stays alive while the retrieved symbol is used below
        let get_api_base: libloading::Symbol<
            unsafe extern "C" fn() -> *const ort::sys::OrtApiBase,
        > = unsafe { library.get(b"OrtGetApiBase") }.map_err(|_| {
            DynamicRuntimeError::MissingApiBase {
                path: path.to_path_buf(),
            }
        })?;

        // safety: `OrtGetApiBase` has the stable ONNX Runtime entrypoint signature
        let api_base = unsafe { get_api_base() };
        if api_base.is_null() {
            return Err(DynamicRuntimeError::NullApiBase {
                path: path.to_path_buf(),
            });
        }

        // safety: the validated runtime exposes a process-stable version string pointer
        let version_ptr = unsafe { ((*api_base).GetVersionString)() };
        // safety: ONNX Runtime documents the version string as a null-terminated C string
        let version = unsafe { CStr::from_ptr(version_ptr) }
            .to_string_lossy()
            .into_owned();
        let minor = version
            .split('.')
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if minor < ort::MINOR_VERSION {
            return Err(DynamicRuntimeError::IncompatibleVersion {
                path: path.to_path_buf(),
                required_minor: ort::MINOR_VERSION,
                found_version: version,
            });
        }

        Ok(())
    }

    const fn default_library_name() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "onnxruntime.dll"
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            "libonnxruntime.so"
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            "libonnxruntime.dylib"
        }
    }

    fn format_paths(paths: &[PathBuf]) -> String {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::with_capacity(paths.len());
    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    /// Empty names no device; anything else is left to ONNX Runtime, whose set of legal
    /// strings is larger than this crate's and still growing.
    #[test]
    fn only_an_empty_device_string_is_refused() {
        for device in ["", " ", "\t"] {
            assert!(
                ExecutionMode::OpenVino {
                    device_type: device
                }
                .validate()
                .is_err(),
                "{device:?} names no device",
            );
        }

        #[cfg(feature = "openvino")]
        for device in ["GPU", "AUTO", "BATCH:GPU(4)", "FUTURE:GPU", "gpu"] {
            assert!(
                ExecutionMode::OpenVino {
                    device_type: device
                }
                .validate()
                .is_ok(),
                "{device:?} is ONNX Runtime's to accept or refuse, not this crate's",
            );
        }
    }

    /// 🔴 The point is the second call. A runtime device string has to become `&'static str`
    /// somehow, and every consumer that did it by hand leaked one per call site; interning
    /// bounds the total by how many different devices a process asks for, which is one or two.
    #[test]
    fn the_same_device_string_is_leaked_once() {
        let a = ExecutionMode::openvino("GPU.1");
        let b = ExecutionMode::openvino(&String::from("GPU.1"));
        match (a, b) {
            (
                ExecutionMode::OpenVino { device_type: x },
                ExecutionMode::OpenVino { device_type: y },
            ) => {
                assert_eq!(x, "GPU.1");
                assert!(std::ptr::eq(x, y), "asking twice must not leak twice");
            }
            other => panic!("{other:?} is not an OpenVINO mode"),
        }
    }

    /// A log line wants the device; `as_str` cannot carry it and must not start to.
    #[test]
    fn the_label_carries_the_device_and_as_str_does_not() {
        let mode = ExecutionMode::openvino("GPU.1");
        assert_eq!(mode.label(), "openvino:GPU.1");
        assert_eq!(mode.as_str(), "openvino");
        assert_eq!(ExecutionMode::Cuda.label(), "cuda");
        assert_eq!(format!("{}", ExecutionMode::Cuda), "cuda");
    }

    #[cfg(any(
        not(feature = "coreml"),
        not(feature = "cuda"),
        not(feature = "migraphx"),
        not(feature = "openvino")
    ))]
    use super::ExecutionMode;
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    use super::{DynamicRuntimeError, OrtRuntimeError, ensure_ort_ready};

    #[cfg(not(feature = "coreml"))]
    #[test]
    fn coreml_modes_require_feature() {
        let error = ExecutionMode::CoreMl.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml requires the `coreml` Cargo feature"
        );

        let error = ExecutionMode::CoreMlFast.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml-fast requires the `coreml` Cargo feature"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_modes_require_feature() {
        let error = ExecutionMode::Cuda.validate().unwrap_err();
        assert_eq!(error.to_string(), "cuda requires the `cuda` Cargo feature");

        let error = ExecutionMode::CudaFast.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "cuda-fast requires the `cuda` Cargo feature"
        );
    }

    #[cfg(not(feature = "migraphx"))]
    #[test]
    fn migraphx_mode_requires_feature() {
        let error = ExecutionMode::MiGraphX.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "migraphx requires the `migraphx` Cargo feature"
        );
    }

    #[cfg(not(feature = "openvino"))]
    #[test]
    fn openvino_mode_requires_feature() {
        let error = ExecutionMode::OpenVino { device_type: "GPU" }
            .validate()
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "openvino requires the `openvino` Cargo feature"
        );
    }

    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    #[test]
    fn dynamic_runtime_preflight_fails_instead_of_hanging() {
        let original = std::env::var_os("ORT_DYLIB_PATH");
        let missing = std::env::temp_dir().join("missing-ort-runtime/libonnxruntime.dylib");
        // safety: this test mutates a process-global env var and restores it before returning
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &missing);
        }

        let error = ensure_ort_ready().unwrap_err();
        assert!(matches!(
            error,
            super::ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
                DynamicRuntimeError::Missing { .. }
            ))
        ));

        // safety: this test restores the original process-global env var before returning
        unsafe {
            match original {
                Some(value) => std::env::set_var("ORT_DYLIB_PATH", value),
                None => std::env::remove_var("ORT_DYLIB_PATH"),
            }
        }
    }
}
