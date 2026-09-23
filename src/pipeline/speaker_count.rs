//! Constraining how many speakers a run reports, the way pyannote's `num_speakers`,
//! `min_speakers` and `max_speakers` do.
//!
//! 🔴 **Status: parked, untested end to end -- do not merge.** Only the unit tests below have
//! run. No real recording has been through these call sites, the engine and the shell do not
//! pass a count yet, and the regression and accuracy checks have not been done. What remains,
//! and when to pick it up, is in `SPEAKER-COUNT.md` in beclab/speakrs-diarization.
//!
//! Its own file for the reason `FORK.md` gives. What stays in upstream's files is a config field
//! and two call sites: `TrainingEmbeddings::cluster` hands its centroids to [`assign`], and
//! `post_inference` caps the per-frame speaker count with [`SpeakerCount::cap`].
//!
//! Three pieces, and all three are needed. Re-clustering alone is not enough: with the count
//! forced below what segmentation sees in a chunk, the constrained assignment leaves a local
//! speaker without a cluster and its speech is dropped, and an uncapped per-frame count makes
//! reconstruction pick a zero-padded column -- a speaker nobody asked for.

use ndarray::{Array2, Axis, s};

use super::clustering::assign_chunk_embeddings;
use super::types::{ChunkEmbeddings, DecodedSegmentations, SpeakerCountTrack};
use crate::utils::cosine_similarity;

/// Bounds on the number of speakers a run may report.
///
/// The default constrains nothing, and a run under it is the unconstrained pipeline, bit for
/// bit. A zero in any field means the same as `None`, as it does in pyannote.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeakerCount {
    /// Exactly this many speakers. Overrides `min` and `max`.
    pub num: Option<usize>,
    /// At least this many.
    pub min: Option<usize>,
    /// At most this many.
    pub max: Option<usize>,
}

impl SpeakerCount {
    /// Exactly `num` speakers.
    pub fn exactly(num: usize) -> Self {
        Self {
            num: Some(num),
            ..Self::default()
        }
    }

    /// Whether these bounds constrain anything.
    pub fn is_unconstrained(&self) -> bool {
        self.bounds() == (1, usize::MAX)
    }

    /// Rejects bounds no output can satisfy. pyannote raises on the same condition.
    pub fn validate(&self) -> Result<(), String> {
        let (min, max) = self.bounds();
        if min > max {
            return Err(format!(
                "min_speakers ({min}) must not exceed max_speakers ({max})"
            ));
        }
        Ok(())
    }

    /// pyannote's `set_num_speakers`: `num` overrides both, `min` defaults to 1, `max` to
    /// unbounded.
    fn bounds(&self) -> (usize, usize) {
        let given = |value: Option<usize>| value.filter(|count| *count > 0);
        let num = given(self.num);
        let min = num.or(given(self.min)).unwrap_or(1);
        let max = num.or(given(self.max)).unwrap_or(usize::MAX);
        (min, max)
    }

    /// Caps the per-frame count of simultaneous speakers at `max`.
    ///
    /// Segmentation can overcount, and reconstruction activates as many clusters per frame as
    /// this track says. With fewer clusters than that, it pads with zero columns and activates
    /// those -- speakers that do not exist.
    pub(crate) fn cap(&self, track: SpeakerCountTrack) -> SpeakerCountTrack {
        let (_, max) = self.bounds();
        if max == usize::MAX {
            return track;
        }
        SpeakerCountTrack(track.0.into_iter().map(|count| count.min(max)).collect())
    }

    /// The cluster count VBx has to be corrected to, if any. `None` when `auto` already
    /// satisfies the bounds, which is the only answer an unconstrained run can get.
    fn correction(&self, auto: usize, num_embeddings: usize) -> Option<usize> {
        let (min, max) = self.bounds();
        let target = if auto < min {
            min
        } else if auto > max {
            max
        } else {
            self.num.filter(|count| *count > 0)?
        };
        // KMeans cannot make more clusters than there are embeddings to put in them.
        let target = target.clamp(1, num_embeddings.max(1));
        (target != auto).then_some(target)
    }
}

/// Assigns every chunk's local speakers to clusters, re-clustering first when VBx's count
/// breaks the bounds.
///
/// Unconstrained, or constrained and already satisfied, this is `assign_chunk_embeddings`
/// with VBx's centroids, unchanged. Otherwise it follows pyannote's `VBxClustering`: KMeans on
/// the L2-normalised training embeddings, centroids as the mean of the raw ones, and each local
/// speaker to its nearest centroid with no one-cluster-per-speaker constraint -- because under a
/// forced count two local speakers in one chunk are often the same person, and the constraint
/// would leave one of them, and its speech, unassigned.
pub(super) fn assign(
    segmentations: &DecodedSegmentations,
    embeddings: &ChunkEmbeddings,
    training: &Array2<f32>,
    centroids: Array2<f32>,
    bounds: &SpeakerCount,
) -> Array2<i32> {
    let Some(target) = bounds.correction(centroids.nrows(), training.nrows()) else {
        return assign_chunk_embeddings(segmentations, embeddings, &centroids);
    };
    let labels = kmeans(
        &l2_normalized_rows(training),
        target,
        KMEANS_RUNS,
        KMEANS_SEED,
    );
    let centroids = cluster_means(training, &labels, target);
    assign_unconstrained(segmentations, embeddings, &centroids)
}

// sklearn's KMeans as pyannote calls it: n_init=3, random_state=42. The seed makes a run
// repeatable; it does not make it match sklearn's numbers, which no reimplementation can.
const KMEANS_RUNS: usize = 3;
const KMEANS_SEED: u64 = 42;
const KMEANS_MAX_ITERS: usize = 300;

fn l2_normalized_rows(rows: &Array2<f32>) -> Array2<f32> {
    let mut normalized = rows.clone();
    for mut row in normalized.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|value| value / norm);
        }
    }
    normalized
}

/// Mean of the rows in each cluster. A cluster that ended up empty is left out, so the result
/// can have fewer rows than `k`; pyannote gets a NaN centroid there, which nothing is ever
/// nearest to, so the output is the same.
fn cluster_means(rows: &Array2<f32>, labels: &[usize], k: usize) -> Array2<f32> {
    let mut sums = Array2::<f32>::zeros((k, rows.ncols()));
    let mut counts = vec![0usize; k];
    for (row, &label) in rows.axis_iter(Axis(0)).zip(labels) {
        sums.row_mut(label).scaled_add(1.0, &row);
        counts[label] += 1;
    }
    let kept: Vec<usize> = (0..k).filter(|&cluster| counts[cluster] > 0).collect();
    let mut means = Array2::<f32>::zeros((kept.len(), rows.ncols()));
    for (out, &cluster) in kept.iter().enumerate() {
        means.row_mut(out).assign(
            &sums
                .row(cluster)
                .mapv(|value| value / counts[cluster] as f32),
        );
    }
    means
}

/// Lloyd's algorithm from k-means++ seeds, best of `runs` by inertia.
fn kmeans(points: &Array2<f32>, k: usize, runs: usize, seed: u64) -> Vec<usize> {
    let mut rng = SplitMix64(seed);
    let mut best: Option<(f64, Vec<usize>)> = None;
    for _ in 0..runs.max(1) {
        let mut centers = kmeans_plus_plus(points, k, &mut rng);
        let mut labels = vec![usize::MAX; points.nrows()];
        for _ in 0..KMEANS_MAX_ITERS {
            let mut changed = false;
            for (idx, point) in points.axis_iter(Axis(0)).enumerate() {
                let nearest = nearest_center(&point, &centers).0;
                changed |= labels[idx] != nearest;
                labels[idx] = nearest;
            }
            if !changed {
                break;
            }
            let means = cluster_sums(points, &labels, k);
            for (cluster, (sum, count)) in means.into_iter().enumerate() {
                // An emptied cluster keeps its last center rather than collapsing to the origin.
                if count > 0 {
                    centers
                        .row_mut(cluster)
                        .assign(&sum.mapv(|value| value / count as f32));
                }
            }
        }
        let inertia: f64 = points
            .axis_iter(Axis(0))
            .map(|point| nearest_center(&point, &centers).1)
            .sum();
        if best.as_ref().is_none_or(|(lowest, _)| inertia < *lowest) {
            best = Some((inertia, labels));
        }
    }
    best.map(|(_, labels)| labels).unwrap_or_default()
}

fn cluster_sums(
    points: &Array2<f32>,
    labels: &[usize],
    k: usize,
) -> Vec<(ndarray::Array1<f32>, usize)> {
    let mut sums = vec![(ndarray::Array1::<f32>::zeros(points.ncols()), 0usize); k];
    for (point, &label) in points.axis_iter(Axis(0)).zip(labels) {
        sums[label].0.scaled_add(1.0, &point);
        sums[label].1 += 1;
    }
    sums
}

fn kmeans_plus_plus(points: &Array2<f32>, k: usize, rng: &mut SplitMix64) -> Array2<f32> {
    let n = points.nrows();
    let mut centers = Array2::<f32>::zeros((k, points.ncols()));
    let first = rng.below(n);
    centers.row_mut(0).assign(&points.row(first));
    let mut nearest: Vec<f64> = points
        .axis_iter(Axis(0))
        .map(|point| squared_distance(&point, &centers.row(0)))
        .collect();
    for cluster in 1..k {
        let total: f64 = nearest.iter().sum();
        let pick = if total > 0.0 {
            let mut target = rng.unit() * total;
            let mut chosen = n - 1;
            for (idx, weight) in nearest.iter().enumerate() {
                if target < *weight {
                    chosen = idx;
                    break;
                }
                target -= weight;
            }
            chosen
        } else {
            // Every point sits on a center already: any pick is as good as another.
            rng.below(n)
        };
        centers.row_mut(cluster).assign(&points.row(pick));
        for (idx, point) in points.axis_iter(Axis(0)).enumerate() {
            nearest[idx] = nearest[idx].min(squared_distance(&point, &centers.row(cluster)));
        }
    }
    centers
}

fn nearest_center(point: &ndarray::ArrayView1<f32>, centers: &Array2<f32>) -> (usize, f64) {
    centers
        .axis_iter(Axis(0))
        .map(|center| squared_distance(point, &center))
        .enumerate()
        .fold((0, f64::INFINITY), |best, (idx, distance)| {
            if distance < best.1 {
                (idx, distance)
            } else {
                best
            }
        })
}

fn squared_distance(lhs: &ndarray::ArrayView1<f32>, rhs: &ndarray::ArrayView1<f32>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| {
            let diff = f64::from(*a) - f64::from(*b);
            diff * diff
        })
        .sum()
}

/// Each active local speaker to its most similar centroid, clusters shared freely.
///
/// An embedding that is not finite scores no cluster; pyannote's `argmax` over its NaN row
/// returns the first cluster, and so does this.
fn assign_unconstrained(
    segmentations: &DecodedSegmentations,
    embeddings: &ChunkEmbeddings,
    centroids: &Array2<f32>,
) -> Array2<i32> {
    let num_chunks = embeddings.0.shape()[0];
    let num_speakers = embeddings.0.shape()[1];
    let mut labels = Array2::<i32>::from_elem((num_chunks, num_speakers), -2);
    for chunk_idx in 0..num_chunks {
        for speaker_idx in 0..num_speakers {
            let active = segmentations.0.slice(s![chunk_idx, .., speaker_idx]).sum() > 0.0;
            if !active || centroids.nrows() == 0 {
                continue;
            }
            let embedding = embeddings.0.slice(s![chunk_idx, speaker_idx, ..]);
            let mut best = (0usize, f32::NEG_INFINITY);
            if embedding.iter().all(|value| value.is_finite()) {
                for (cluster_idx, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
                    let score = cosine_similarity(&embedding, &centroid);
                    if score > best.1 {
                        best = (cluster_idx, score);
                    }
                }
            }
            labels[[chunk_idx, speaker_idx]] = best.0 as i32;
        }
    }
    labels
}

/// SplitMix64: small, fixed, and enough for seeding KMeans. A dependency for this would be a
/// crate the engine's lockfile has to carry for twenty lines.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize % n.max(1)
    }
}

#[cfg(test)]
#[path = "speaker_count_tests.rs"]
mod tests;
