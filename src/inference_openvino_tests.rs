//! What an OpenVINO device string means to this crate, and what a build without the
//! `openvino` feature refuses.
//!
//! Kept out of the shared `mod tests` block so that upstream's test additions and ours
//! never land on the same lines -- that block is now byte-identical to upstream's.`FORK.md` says why, once.
//!
//! The import below is unconditional on purpose: the tests asserting what an empty device
//! string does run on every build, so guarding it the way the shared block guards its own
//! made the import vanish whenever every accelerated feature was on, and this file stopped
//! compiling.

use super::{ExecutionMode, names_a_device};

/// Empty names no device; anything else is left to ONNX Runtime, whose set of legal
/// strings is larger than this crate's and still growing.
///
/// The second half asks the predicate rather than `validate`, so it runs on every build.
/// Through `validate` it needed the `openvino` feature, and no job in this repository's CI
/// builds that -- so this test was green on the strength of its first half, and the "only"
/// in its name was exactly the part nothing ran.
#[test]
fn only_an_empty_device_string_is_refused() {
    for device in ["", " ", "\t"] {
        assert!(!names_a_device(device), "{device:?} names no device");
        assert!(
            ExecutionMode::OpenVino {
                device_type: device
            }
            .validate()
            .is_err(),
            "{device:?} names no device",
        );
    }

    for device in ["GPU", "AUTO", "BATCH:GPU(4)", "FUTURE:GPU", "gpu"] {
        assert!(
            names_a_device(device),
            "{device:?} is ONNX Runtime's to accept or refuse, not this crate's",
        );
    }
}

/// Which of the two refusals it is has to survive into the message. The empty string is
/// caught before the feature is looked at, so this is what a build with `openvino` sees
/// too -- and naming the feature there would be false, since it is enabled.
#[test]
fn an_empty_device_string_is_not_blamed_on_a_missing_feature() {
    let error = ExecutionMode::OpenVino { device_type: "" }
        .validate()
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "openvino was given a device string naming no device",
    );
}

/// The point is the second call. A runtime device string has to become `&'static str`
/// somehow, and doing that by hand leaks one per call site; interning
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
