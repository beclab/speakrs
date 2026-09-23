//! Speaker-count bounds: what they resolve to, and what each of the three pieces changes.
//!
//! Kept out of the shared `mod tests` block; `FORK.md` says why, once.

use ndarray::{Array2, Array3, array, s};

use super::*;

/// Two chunks, three local speakers, 2-d embeddings on two well-separated directions.
/// Chunk 0 has local speakers 0 and 1 active, on different voices; chunk 1 has 0 and 1 active
/// on the same voice as each other -- the case a forced single speaker has to survive.
fn two_chunk_fixture() -> (DecodedSegmentations, ChunkEmbeddings) {
    let mut segmentations = Array3::<f32>::zeros((2, 4, 3));
    segmentations.slice_mut(s![.., .., 0]).fill(1.0);
    segmentations.slice_mut(s![.., .., 1]).fill(1.0);
    let mut embeddings = Array3::<f32>::zeros((2, 3, 2));
    embeddings.slice_mut(s![0, 0, ..]).assign(&array![1.0, 0.0]);
    embeddings.slice_mut(s![0, 1, ..]).assign(&array![0.0, 1.0]);
    embeddings.slice_mut(s![1, 0, ..]).assign(&array![1.0, 0.1]);
    embeddings
        .slice_mut(s![1, 1, ..])
        .assign(&array![1.0, -0.1]);
    embeddings.slice_mut(s![.., 2, ..]).fill(f32::NAN);
    (
        DecodedSegmentations(segmentations),
        ChunkEmbeddings(embeddings),
    )
}

fn training_rows() -> Array2<f32> {
    array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [1.0, -0.1]]
}

#[test]
fn default_constrains_nothing() {
    assert!(SpeakerCount::default().is_unconstrained());
    assert!(!SpeakerCount::exactly(1).is_unconstrained());
}

#[test]
fn zero_means_not_given() {
    let zeros = SpeakerCount {
        num: Some(0),
        min: Some(0),
        max: Some(0),
    };
    assert!(zeros.is_unconstrained());
}

#[test]
fn num_overrides_min_and_max() {
    let bounds = SpeakerCount {
        num: Some(2),
        min: Some(5),
        max: Some(9),
    };
    assert_eq!(bounds.bounds(), (2, 2));
}

#[test]
fn min_above_max_is_rejected() {
    let bounds = SpeakerCount {
        num: None,
        min: Some(4),
        max: Some(2),
    };
    assert!(bounds.validate().is_err());
    assert!(SpeakerCount::exactly(3).validate().is_ok());
}

#[test]
fn cap_leaves_an_unbounded_track_alone_and_clips_a_bounded_one() {
    let track = SpeakerCountTrack(vec![0, 1, 2, 3]);
    assert_eq!(
        SpeakerCount::default().cap(track.clone()).0,
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        SpeakerCount::exactly(1).cap(track.clone()).0,
        vec![0, 1, 1, 1]
    );
    let at_most_two = SpeakerCount {
        max: Some(2),
        ..SpeakerCount::default()
    };
    assert_eq!(at_most_two.cap(track).0, vec![0, 1, 2, 2]);
}

#[test]
fn correction_only_when_the_bounds_are_broken() {
    assert_eq!(SpeakerCount::default().correction(7, 100), None);
    assert_eq!(SpeakerCount::exactly(2).correction(2, 100), None);
    assert_eq!(SpeakerCount::exactly(1).correction(2, 100), Some(1));
    assert_eq!(SpeakerCount::exactly(4).correction(2, 100), Some(4));
    let between = SpeakerCount {
        min: Some(3),
        max: Some(5),
        ..SpeakerCount::default()
    };
    assert_eq!(between.correction(4, 100), None);
    assert_eq!(between.correction(2, 100), Some(3));
    assert_eq!(between.correction(8, 100), Some(5));
}

#[test]
fn correction_never_asks_for_more_clusters_than_embeddings() {
    assert_eq!(SpeakerCount::exactly(6).correction(2, 3), Some(3));
    assert_eq!(SpeakerCount::exactly(6).correction(3, 3), None);
}

#[test]
fn kmeans_separates_two_blobs_and_repeats_itself() {
    let points = array![
        [1.0, 0.0],
        [0.99, 0.05],
        [0.98, -0.05],
        [0.0, 1.0],
        [0.05, 0.99],
        [-0.05, 0.98]
    ];
    let first = kmeans(&points, 2, 3, 42);
    assert_eq!(first, kmeans(&points, 2, 3, 42));
    assert_eq!(first[0], first[1]);
    assert_eq!(first[1], first[2]);
    assert_eq!(first[3], first[4]);
    assert_eq!(first[4], first[5]);
    assert_ne!(first[0], first[3]);
}

#[test]
fn cluster_means_drops_an_empty_cluster() {
    let rows = array![[1.0, 0.0], [3.0, 0.0]];
    let means = cluster_means(&rows, &[0, 0], 2);
    assert_eq!(means, array![[2.0, 0.0]]);
}

#[test]
fn unconstrained_assignment_is_upstreams() {
    let (segmentations, embeddings) = two_chunk_fixture();
    let centroids = array![[1.0, 0.0], [0.0, 1.0]];
    let ours = assign(
        &segmentations,
        &embeddings,
        &training_rows(),
        centroids.clone(),
        &SpeakerCount::default(),
    );
    assert_eq!(
        ours,
        assign_chunk_embeddings(&segmentations, &embeddings, &centroids)
    );
}

#[test]
fn a_forced_single_speaker_keeps_both_local_speakers_of_a_chunk() {
    let (segmentations, embeddings) = two_chunk_fixture();
    let vbx_found_two = array![[1.0, 0.0], [0.0, 1.0]];
    let labels = assign(
        &segmentations,
        &embeddings,
        &training_rows(),
        vbx_found_two,
        &SpeakerCount::exactly(1),
    );
    // Every active local speaker lands in the one cluster; none is left unassigned (-2), which
    // is what the one-cluster-per-speaker assignment would do to the second of each chunk.
    assert_eq!(labels.slice(s![.., 0..2]), array![[0, 0], [0, 0]]);
    assert_eq!(labels.slice(s![.., 2]), array![-2, -2]);
}

#[test]
fn a_forced_larger_count_splits_what_vbx_merged() {
    let (segmentations, embeddings) = two_chunk_fixture();
    let vbx_found_one = array![[0.7, 0.7]];
    let labels = assign(
        &segmentations,
        &embeddings,
        &training_rows(),
        vbx_found_one,
        &SpeakerCount::exactly(2),
    );
    assert_ne!(labels[[0, 0]], labels[[0, 1]]);
    assert_eq!(labels[[1, 0]], labels[[0, 0]]);
    assert_eq!(labels[[1, 1]], labels[[0, 0]]);
}
