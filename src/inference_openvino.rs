//! What an OpenVINO device string means to this crate: the one place `AUTO`, `MULTI`,
//! `HETERO`, `BATCH` and a device index are interpreted.
//!
//! Its own file so that upstream's edits to `inference.rs` and ours do not land on the same
//! lines. `FORK.md` says why, once. Upstream has changed that file four times since the fork point; this fork adds a
//! backend to it, and the two only have to meet at the enum variant and its match arms.

use super::ExecutionMode;

/// What an OpenVINO device string resolves to, as far as decisions in this crate are concerned.
///
/// The split is by decision, not by kind of hardware. `HETERO:NPU,GPU` and `MULTI:CPU,NPU`
/// are both "several devices", and they want opposite answers: the first reaches the GPU
/// plugin and the second cannot. What every caller here actually asks is whether the GPU
/// plugin is in play.
///
/// It cannot tell a discrete card from an integrated one. `GPU`, `GPU.0` and `GPU.1` are
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
    /// Every name read, not merely no GPU found. A list holding a name this crate cannot
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

/// Whether a device string names a device at all.
///
/// The one thing this crate refuses on its own. Everything else is ONNX Runtime's to accept:
/// the set of legal names lives in its C++ side and in OpenVINO, and it grows. Empty is
/// different -- it names nothing, and the provider's answer to it surfaces a long way from
/// the call that caused it.
///
/// A predicate rather than an expression inside `validate`, so that what it accepts can be
/// asserted on any build. `validate` cannot: without the `openvino` feature every OpenVINO
/// mode is refused for the feature, so the test that the other strings get through only ever
/// ran where the feature was on -- which is no CI job here.
pub(crate) fn names_a_device(device: &str) -> bool {
    !device.trim().is_empty()
}

/// The device names this crate recognises inside a device string.
pub(crate) fn ov_device_token(token: &str) -> Option<&'static str> {
    // The batch size comes off first. `BATCH:GPU(4)` sets the batch explicitly -- OpenVINO's
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
                // `NonGpuComposite` is claimed only when every name in the list was read.
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
/// Bare `AUTO` is deliberately not here, and `may_reach_openvino_gpu` is the one that says
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
