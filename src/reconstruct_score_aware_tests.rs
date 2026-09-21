//! How the score-aware exclusive reconstruction picks a speaker, and what it does on a tie.
//!
//! Kept out of the shared `mod tests` block so that upstream's test additions and ours
//! never land on the same lines.

use ndarray::{Array2, array};

use super::*;
use crate::pipeline::ExclusiveFrameDecision;

#[test]
fn score_aware_exclusive_uses_unique_support() {
    let support = array![[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]];
    let ordinary = Array2::from_elem((3, 2), 1.0);

    let result = score_aware_exclusive(&support, &ordinary);

    assert_eq!(
        &*result.diarization,
        &array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    );
    assert!(
        result
            .frames
            .iter()
            .all(|frame| { frame.decision == Some(ExclusiveFrameDecision::UniqueSupport) })
    );
}

#[test]
fn score_aware_exclusive_resolves_equal_support_from_both_anchors() {
    let support = array![[3.0, 1.0], [2.0, 2.0], [3.0, 1.0]];
    let ordinary = Array2::from_elem((3, 2), 1.0);

    let result = score_aware_exclusive(&support, &ordinary);

    assert_eq!(
        &*result.diarization,
        &array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    );
    assert_eq!(
        result.frames[1].decision,
        Some(ExclusiveFrameDecision::TieContinuity)
    );
}

#[test]
fn score_aware_exclusive_marks_unanchored_tie_as_fallback() {
    let support = array![[2.0, 2.0], [2.0, 2.0]];
    let ordinary = Array2::from_elem((2, 2), 1.0);

    let result = score_aware_exclusive(&support, &ordinary);

    assert_eq!(&*result.diarization, &array![[1.0, 0.0], [1.0, 0.0]]);
    assert!(
        result
            .frames
            .iter()
            .all(|frame| { frame.decision == Some(ExclusiveFrameDecision::DeterministicFallback) })
    );
}

#[test]
fn score_aware_exclusive_is_invariant_to_cluster_permutation() {
    let support = array![[3.0, 1.0], [2.0, 2.0], [3.0, 1.0]];
    let swapped_support = array![[1.0, 3.0], [2.0, 2.0], [1.0, 3.0]];
    let ordinary = Array2::from_elem((3, 2), 1.0);

    let original = score_aware_exclusive(&support, &ordinary);
    let swapped = score_aware_exclusive(&swapped_support, &ordinary);

    for frame_idx in 0..3 {
        assert_eq!(
            original.frames[frame_idx].speaker_idx,
            swapped.frames[frame_idx].speaker_idx.map(|idx| 1 - idx)
        );
    }
}
