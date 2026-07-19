# JAX Supervised Training — Aircraft Detection & Classification

A JAX/Flax computer vision pipeline that detects military aircraft in video/images and classifies their type (32 aircraft models), built entirely on a JIT-compiled, single-pass inference graph.

![Demo — 4-quadrant live output](docs/demo.gif)

*Top-left: source frame. Top-right: detections (boxes + class) overlaid on a synthetic confidence heatmap. Bottom-left: standalone heatmap — including "near-miss" candidates the model considered but didn't confirm. Bottom-right: classified crops of each confirmed detection.*

## The JAX Single-Pass principle

Most detect-then-classify pipelines are two separate models glued together by host-side (Python/OpenCV) code: run the detector, pull boxes back to the CPU, crop the image, run the classifier on each crop. Every round-trip between GPU and host is dead time, and CPU-side logic (like `cv2.findContours` on a segmentation mask) can't be JIT-compiled or batched cleanly — it also has structural failure modes, e.g. two nearby objects silently merging into one detected blob.

**Single-Pass** fuses the entire pipeline — resize → anchor-free (CenterNet-style) detection → top-K peak extraction → differentiable crop → classification — into **one JIT-compiled, `vmap`-batched JAX function**. A frame goes in, and `{boxes, classes, class_scores, detection_scores, valid_mask}` comes out (20 fixed slots, `valid_mask` as sole authority for what's a real detection), with:

- **No host round-trips** on the critical path — detection and classification live in the same compiled graph, cropping happens on-device via a differentiable resample instead of CPU slicing.
- **Per-instance detection**, not blob extraction — a center-point heatmap head predicts independent objects even when they're close together, structurally avoiding the box-fusion failure mode of the old segmentation+contours approach.
- **Batched, not looped** — a full batch of frames is processed through `jax.vmap` in one dispatch, keeping the GPU fed instead of issuing one call per frame.

The video driver (`bounding_boxes_with_classification_from_video_generation.py`) additionally pipelines GPU inference and CPU rendering across batches (separate threads, bounded queue) to keep both busy concurrently rather than sequentially.

## Stack

JAX / Flax, OpenCV, trained checkpoints for detection (`JAX_DETECTOR`, CenterNet-style) and classification (`FIGHTERJET_CLASSIFICATION`), with tooling to build datasets, train, and run real-time-ish inference on video or image batches.
