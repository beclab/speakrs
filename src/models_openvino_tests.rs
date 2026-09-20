//! Which model files an OpenVINO deployment has to be given.
//!
//! Kept out of the shared `mod tests` block so that upstream's test additions and ours
//! never land on the same lines.

use super::required_files;
use crate::inference::ExecutionMode;

#[test]
fn openvino_required_files_are_the_accelerated_set() {
    let openvino = required_files(ExecutionMode::OpenVino { device_type: "GPU" });

    // Named rather than only compared as a whole, because this one file is the one a
    // reader expects to be missing on the device asked for here: the GPU plugin cannot
    // compile it. It is fetched anyway, and for the GPU it is what the `-dynseq`
    // derivative the load site looks for is made from -- without it that derivative
    // cannot exist, so batching would be off for good. The list is the same for every
    // OpenVINO device, and on the processor and the NPU this file is loaded directly.
    assert!(openvino.contains(&"segmentation-3.0-b32.onnx".to_string()));
    assert!(openvino.contains(&"segmentation-3.0.onnx".to_string()));

    // And no other divergence has crept in: the kernel the GPU plugin trips over is an
    // LSTM one, only segmentation has an LSTM, and that is handled where sessions are
    // built.
    assert_eq!(openvino, required_files(ExecutionMode::MiGraphX));
}
