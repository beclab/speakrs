# What this fork changes, and how to add to it without paying for it twice

This is a fork of [avencera/speakrs](https://github.com/avencera/speakrs). It exists for one
reason: the published library has backends for NVIDIA, Apple and AMD, and none for Intel. The
OpenVINO backend here is that entry. Everything else in this file is about keeping the fork
cheap to carry until the backend lands upstream or the fork is retired.

## The rule

**Put this fork's code where upstream does not edit.** A change that only adds lines, in a file
or at a line range upstream leaves alone, merges by itself. A change interleaved with upstream's
own code has to be merged by hand, every time, forever.

In practice:

- **Tests go in their own file**, reached by `#[cfg(test)] #[path = "..."] mod ...;`. Upstream
  appends tests to the shared `mod tests` block, and so did this fork, so both sides were
  writing at the same line numbers. That one habit produced about 90% of the conflict volume
  before it was undone.
- **Implementations go in their own file** when they are more than a few lines --
  `reconstruct_score_aware.rs`, `inference_openvino.rs`, `segmentation_openvino.rs`,
  `pipeline/types/score_aware.rs`. What has to stay behind is the call site and the enum
  variant, which cannot live anywhere else.
- **Re-export on a line of its own.** Adding a name to a `pub use upstream::{...}` list makes
  that list conflict; a second `pub use` next to it does not.
- 🔴 **The module declaration goes under the imports, not at the end of the file** -- unless
  upstream already had a test module in that file at the fork point. Upstream adds new test
  modules at the tail, so the tail is contested exactly where it looks empty.
- **When upstream deletes something this fork also wants gone, delete the same lines** -- git
  merges two identical deletions as one, while a deletion against a modification conflicts. Worth
  what it costs and no more: measured on this tree, deleting `make_exclusive` ahead of the merge
  that replaces it saves 16 conflict lines and leaves the library with no plain per-frame
  reconstruction in the meantime, so it waits for that merge instead.

## Measuring it

The cost of the fork is one number, and it can be read without merging anything:

```sh
git fetch upstream
git merge-tree --write-tree beclab-master upstream/master
```

The first line is a tree; everything after it names the conflicting files. Count the lines
between `<<<<<<<` and `>>>>>>>` in those blobs for the size. It was 1059 before the moves above
and 265 after.

## What is deliberately not here

The plan for merging upstream, the state of each open question, and what the engine that
consumes this library still has to change are process, not contribution rules, and live with
the team that runs this line rather than in the repository.
