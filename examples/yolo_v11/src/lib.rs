//! yolo11n as pure logical ops — see model.rs for the anatomy and the
//! per-scale DFL respelling that removes the concat-of-slices
//! divergence road; main.rs carries the host letterbox/NMS pipeline
//! and the (documented-heavy) reference search/execute path.

pub mod model;
