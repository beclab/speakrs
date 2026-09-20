//! Which batched segmentation export each OpenVINO device can actually load, and who is
//! allowed to carry on without one.
//!
//! Its own file so that upstream's edits to `segmentation.rs` and ours do not land on the
//! same lines. What has to stay there is the call site inside `with_mode`.

use std::path::{Path, PathBuf};

use crate::inference::{ExecutionMode, OvTarget, openvino_gpu_plugin, ov_target};
use crate::models::SEGMENTATION_ONNX;

use super::{PRIMARY_BATCH_SIZE, batched_model_path};

/// The batched segmentation model this mode should load, if any.
///
/// On OpenVINO this is a different file, not the stock `-b32` export, and the decision
/// belongs here rather than only in `required_files`. That list is behind the `online`
/// feature, so a consumer that downloads weights some other way -- which is every consumer
/// that turns default features off -- has the whole repository in its cache and arrives here
/// with the stock file present. Deciding this in the download list would decide nothing for
/// them.
///
/// What it avoids: OpenVINO's GPU plugin generates an LSTM kernel referencing
/// OUTPUT1_GET_INDEX and OUTPUT2_GET_INDEX without defining them, the OpenCL compiler rejects
/// the program, and the session fails to build -- taking the pipeline with it. Measured on
/// Arrow Lake-S / OpenVINO 2025.4.1, on models verified bit-identical to their originals on
/// CPU first, the trigger is a static sequence length rather than the batch: batch 1 made
/// static fails the same way, batch 32 with only its batch dimension made dynamic still
/// fails, and batch 32 with a dynamic sequence length compiles and runs. The unbatched model
/// survives because its sample count is dynamic, not because it is smaller.
pub(super) fn primary_batched_path(model_path: &Path, mode: ExecutionMode) -> Option<PathBuf> {
    if openvino_gpu_plugin(mode) {
        return dynamic_sequence_batched_path(model_path).filter(|path| path.exists());
    }
    batched_model_path(model_path, PRIMARY_BATCH_SIZE).filter(|path| path.exists())
}

/// Whether a batched segmentation model that is present and will not build costs this mode
/// only its batching, rather than its load.
///
/// True for OpenVINO alone, because there the file is written by the consumer at startup
/// from whatever export is in the cache: a new export shape can pass onnx's checker and still
/// be refused by the plugin, and propagating that stops every installation that derives it
/// the next time the weights change, over a feature whose absence costs speed. Every other
/// backend gets the export with its weights, so present-and-unbuildable is a damaged download
/// and the load should stop on it -- which is what they did before this backend existed, and
/// what they must keep doing: adding a backend is not a licence to change the others.
///
/// Narrowed to the devices that can actually refuse: OpenVINO on the processor or the NPU
/// loads the stock export that ships with the weights, so present-and-unbuildable is a damaged
/// download there too and the load should stop on it, as it does for CUDA.
///
/// `UnresolvedAuto` is in, and has to be. Bare `AUTO` takes the stock export -- see
/// `openvino_gpu_plugin` for why -- and may still land on a GPU that refuses it. Leaving it out
/// would mean bare `AUTO` cannot start on a machine with an Intel GPU, which is worse than what
/// it does today.
pub(super) fn tolerates_unbuildable_batched(mode: ExecutionMode) -> bool {
    match mode {
        ExecutionMode::OpenVino { device_type } => matches!(
            ov_target(device_type),
            OvTarget::GpuOnly | OvTarget::GpuComposite | OvTarget::UnresolvedAuto
        ),
        _ => false,
    }
}

/// The batched segmentation model this mode looks for, which is not the same file for all of
/// them.
///
/// Asked rather than spelled, and asked per mode, because a consumer that reports whether
/// batching took has to name the file it looked for -- and naming the OpenVINO one on a CUDA
/// deployment tells the reader to go find a file nothing there wants. It answers for the mode
/// exactly as `primary_batched_path` decides, so the two cannot drift.
pub fn batched_segmentation_file_name_for(mode: ExecutionMode) -> String {
    if openvino_gpu_plugin(mode) {
        return batched_segmentation_file_name();
    }
    let stem = SEGMENTATION_ONNX
        .strip_suffix(".onnx")
        .expect("SEGMENTATION_ONNX names an .onnx file");
    format!("{stem}-b{PRIMARY_BATCH_SIZE}.onnx")
}

/// The file name OpenVINO's **GPU plugin** looks for when it batches segmentation.
///
/// Not the file every OpenVINO device looks for. On the processor and the NPU the loader takes
/// the stock export that ships with the weights, and `batched_segmentation_file_name_for` is
/// the one that answers per mode. This is the derivative, and the name is the same on the
/// devices that use it as it is on the step that writes it -- which is why it is public.
///
/// Built from SEGMENTATION_ONNX and PRIMARY_BATCH_SIZE rather than spelled out. Anything that
/// spells it out instead -- the step that writes the file, and whatever reports that it is
/// there -- goes quietly wrong the moment either constant changes, in a direction whose only
/// symptom is batching being off.
pub fn batched_segmentation_file_name() -> String {
    let stem = SEGMENTATION_ONNX
        .strip_suffix(".onnx")
        .expect("SEGMENTATION_ONNX names an .onnx file");
    format!("{stem}-b{PRIMARY_BATCH_SIZE}-dynseq.onnx")
}

/// The batched segmentation model OpenVINO can compile, named apart from the stock one.
///
/// A different filename rather than the same one, because the two are not interchangeable
/// and the stock one is what crashes. Sharing the name would mean a deployment without the
/// prepared model silently picking up the static graph and failing to build a session -- the
/// failure this whole path exists to avoid. Absent, batching is simply off, which is where
/// OpenVINO stood before this and is the safe direction to fall.
///
/// The file is the stock `-b32` export with its sample dimension made dynamic, which is the
/// one edit that matters and is numerically identity: verified bit-for-bit against the
/// original on CPU before either was trusted. Producing it needs an ONNX library, so it is
/// generated where one is available rather than here.
fn dynamic_sequence_batched_path(model_path: &Path) -> Option<PathBuf> {
    // The name is fixed rather than derived from what was passed, so that the one string a
    // provisioning step has to produce has exactly one definition. The stem is still checked,
    // because a caller handing this something that is not a model should get None.
    model_path.file_name()?.to_str()?.strip_suffix(".onnx")?;
    Some(model_path.with_file_name(batched_segmentation_file_name()))
}
