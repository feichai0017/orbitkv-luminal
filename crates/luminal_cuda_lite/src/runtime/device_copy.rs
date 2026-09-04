use cudarc::driver::{CudaEvent, result};
use luminal::{op::IntoEgglogOp, prelude::NodeIndex};
use rustc_hash::FxHashSet;

use super::CudaRuntimeImpl;

/// One non-overlapping byte range copied within a runtime-owned tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceCopyRange {
    pub source_offset: usize,
    pub destination_offset: usize,
    pub bytes: usize,
}

/// All byte-range copies applied to one runtime output tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceCopyPlan {
    pub tensor: NodeIndex,
    pub ranges: Box<[DeviceCopyRange]>,
}

/// All byte-range copies applied directly to one installed graph input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInputCopyPlan {
    pub tensor: NodeIndex,
    pub ranges: Box<[DeviceCopyRange]>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum DeviceCopyError {
    EmptyTensorPlan,
    DuplicateTensor,
    UnknownTensor,
    MissingBuffer,
    MissingExternalBuffer,
    EmptyRange,
    RangeOverflow,
    RangeOutOfBounds,
    OverlappingRanges,
    PointerOverflow,
    Driver(cudarc::driver::DriverError),
}

impl DeviceCopyError {
    /// Whether a failed call may already have enqueued a device copy.
    pub const fn may_have_enqueued_work(&self) -> bool {
        matches!(self, Self::Driver(_))
    }
}

impl std::fmt::Display for DeviceCopyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTensorPlan => {
                formatter.write_str("device copy tensor plan must not be empty")
            }
            Self::DuplicateTensor => {
                formatter.write_str("device copy plan contains a duplicate tensor")
            }
            Self::UnknownTensor => formatter.write_str("cannot find output tensor in runtime"),
            Self::MissingBuffer => formatter.write_str("cannot find tensor buffer in runtime"),
            Self::MissingExternalBuffer => {
                formatter.write_str("raw pointer input has no external buffer view")
            }
            Self::EmptyRange => formatter.write_str("device copy range must not be empty"),
            Self::RangeOverflow => formatter.write_str("device copy range overflow"),
            Self::RangeOutOfBounds => formatter.write_str("device copy range exceeds tensor"),
            Self::OverlappingRanges => formatter.write_str("device copy ranges overlap"),
            Self::PointerOverflow => formatter.write_str("device pointer overflow"),
            Self::Driver(error) => write!(formatter, "CUDA device copy failed: {error}"),
        }
    }
}

impl std::error::Error for DeviceCopyError {}

impl From<cudarc::driver::DriverError> for DeviceCopyError {
    fn from(error: cudarc::driver::DriverError) -> Self {
        Self::Driver(error)
    }
}

impl<O: IntoEgglogOp> CudaRuntimeImpl<O> {
    /// Enqueues validated, non-overlapping copies within resolved output
    /// tensors and records an event after the final copy.
    ///
    /// The returned event is the completion boundary. A failed call may have
    /// enqueued an earlier copy, so callers must treat the outcome as observed
    /// and ambiguous rather than retrying the same mutation.
    pub fn copy_output_ranges(
        &self,
        plans: &[DeviceCopyPlan],
    ) -> Result<CudaEvent, DeviceCopyError> {
        let mut seen = FxHashSet::default();
        let copies = plans
            .iter()
            .map(|plan| {
                if !seen.insert(plan.tensor) {
                    return Err(DeviceCopyError::DuplicateTensor);
                }
                let buffer = self.try_resolve_output_buffer(plan.tensor)?;
                if plan.ranges.is_empty() {
                    return Err(DeviceCopyError::EmptyTensorPlan);
                }
                validate_device_copy_ranges(buffer.len(), &plan.ranges)?;
                plan.ranges
                    .iter()
                    .map(|range| {
                        let source_offset = u64::try_from(range.source_offset)
                            .map_err(|_| DeviceCopyError::PointerOverflow)?;
                        let destination_offset = u64::try_from(range.destination_offset)
                            .map_err(|_| DeviceCopyError::PointerOverflow)?;
                        let source = buffer
                            .ptr()
                            .checked_add(source_offset)
                            .ok_or(DeviceCopyError::PointerOverflow)?;
                        let destination = buffer
                            .ptr()
                            .checked_add(destination_offset)
                            .ok_or(DeviceCopyError::PointerOverflow)?;
                        Ok((source, destination, range.bytes))
                    })
                    .collect::<Result<Vec<_>, DeviceCopyError>>()
            })
            .collect::<Result<Vec<_>, DeviceCopyError>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for (source, destination, bytes) in copies {
            unsafe {
                result::memcpy_dtod_async(
                    destination,
                    source,
                    bytes,
                    self.cuda_stream.cu_stream(),
                )?;
            }
        }
        Ok(self.cuda_stream.record_event(None)?)
    }

    /// Enqueues validated, non-overlapping copies within installed graph-input
    /// buffers and records an event after the final copy.
    pub fn copy_input_ranges(
        &self,
        plans: &[DeviceInputCopyPlan],
    ) -> Result<CudaEvent, DeviceCopyError> {
        let mut seen = FxHashSet::default();
        let copies = plans
            .iter()
            .map(|plan| {
                if !seen.insert(plan.tensor) {
                    return Err(DeviceCopyError::DuplicateTensor);
                }
                let (pointer, bytes) = self
                    .current_hlir_device_binding(plan.tensor)
                    .ok_or(DeviceCopyError::MissingBuffer)?;
                if plan.ranges.is_empty() {
                    return Err(DeviceCopyError::EmptyTensorPlan);
                }
                validate_device_copy_ranges(bytes, &plan.ranges)?;
                plan.ranges
                    .iter()
                    .map(|range| {
                        let source_offset = u64::try_from(range.source_offset)
                            .map_err(|_| DeviceCopyError::PointerOverflow)?;
                        let destination_offset = u64::try_from(range.destination_offset)
                            .map_err(|_| DeviceCopyError::PointerOverflow)?;
                        let source = pointer
                            .checked_add(source_offset)
                            .ok_or(DeviceCopyError::PointerOverflow)?;
                        let destination = pointer
                            .checked_add(destination_offset)
                            .ok_or(DeviceCopyError::PointerOverflow)?;
                        Ok((source, destination, range.bytes))
                    })
                    .collect::<Result<Vec<_>, DeviceCopyError>>()
            })
            .collect::<Result<Vec<_>, DeviceCopyError>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for (source, destination, bytes) in copies {
            unsafe {
                result::memcpy_dtod_async(
                    destination,
                    source,
                    bytes,
                    self.cuda_stream.cu_stream(),
                )?;
            }
        }
        Ok(self.cuda_stream.record_event(None)?)
    }
}

fn validate_device_copy_ranges(
    buffer_bytes: usize,
    ranges: &[DeviceCopyRange],
) -> Result<(), DeviceCopyError> {
    let mut occupied = Vec::with_capacity(ranges.len().saturating_mul(2));
    for range in ranges {
        if range.bytes == 0 {
            return Err(DeviceCopyError::EmptyRange);
        }
        let source_end = range
            .source_offset
            .checked_add(range.bytes)
            .ok_or(DeviceCopyError::RangeOverflow)?;
        let destination_end = range
            .destination_offset
            .checked_add(range.bytes)
            .ok_or(DeviceCopyError::RangeOverflow)?;
        if source_end > buffer_bytes || destination_end > buffer_bytes {
            return Err(DeviceCopyError::RangeOutOfBounds);
        }
        occupied.push((range.source_offset, source_end));
        occupied.push((range.destination_offset, destination_end));
    }
    occupied.sort_unstable();
    if occupied.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(DeviceCopyError::OverlappingRanges);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_disjoint_source_and_destination_ranges() {
        assert!(
            validate_device_copy_ranges(
                128,
                &[
                    DeviceCopyRange {
                        source_offset: 0,
                        destination_offset: 64,
                        bytes: 16,
                    },
                    DeviceCopyRange {
                        source_offset: 16,
                        destination_offset: 80,
                        bytes: 16,
                    },
                ]
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_overlapping_or_out_of_bounds_ranges() {
        assert_eq!(
            validate_device_copy_ranges(
                64,
                &[DeviceCopyRange {
                    source_offset: 0,
                    destination_offset: 8,
                    bytes: 16,
                }]
            ),
            Err(DeviceCopyError::OverlappingRanges)
        );
        assert_eq!(
            validate_device_copy_ranges(
                64,
                &[DeviceCopyRange {
                    source_offset: 0,
                    destination_offset: 56,
                    bytes: 16,
                }]
            ),
            Err(DeviceCopyError::RangeOutOfBounds)
        );
    }
}
