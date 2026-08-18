//! Optional CUDA 12.8 batched device-copy entry point.

use cudarc::driver::sys;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

const MEM_LOCATION_TYPE_DEVICE: u32 = 1;

#[repr(C)]
struct MemLocation {
    location_type: u32,
    id: i32,
}

#[repr(C)]
struct MemcpyAttributes {
    src_access_order: u32,
    src_location: MemLocation,
    dst_location: MemLocation,
    flags: u32,
}

type MemcpyBatchAsync = unsafe extern "C" fn(
    *mut sys::CUdeviceptr,
    *mut sys::CUdeviceptr,
    *mut usize,
    usize,
    *mut MemcpyAttributes,
    *mut usize,
    usize,
    *mut usize,
    sys::CUstream,
) -> sys::CUresult;

struct Api {
    _library: libloading::Library,
    memcpy_batch_async: MemcpyBatchAsync,
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| {
        #[cfg(unix)]
        let candidates = ["libcuda.so.1", "libcuda.so"];
        #[cfg(windows)]
        let candidates = ["nvcuda.dll", "nvcuda.dll"];

        candidates.into_iter().find_map(|name| unsafe {
            let library = libloading::Library::new(name).ok()?;
            let memcpy_batch_async = *library
                .get::<MemcpyBatchAsync>(b"cuMemcpyBatchAsync\0")
                .ok()?;
            Some(Api {
                _library: library,
                memcpy_batch_async,
            })
        })
    })
    .as_ref()
}

static DISABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn is_available() -> bool {
    !DISABLED.load(Ordering::Relaxed) && api().is_some()
}

/// Returns `None` when the loaded CUDA driver predates CUDA 12.8.
pub(crate) unsafe fn copy(
    copies: &[(u64, u64, usize)],
    stream: sys::CUstream,
    device_ordinal: i32,
) -> Option<sys::CUresult> {
    if DISABLED.load(Ordering::Relaxed) {
        return None;
    }
    let api = api()?;
    let mut destinations = copies.iter().map(|copy| copy.0).collect::<Vec<_>>();
    let mut sources = copies.iter().map(|copy| copy.1).collect::<Vec<_>>();
    let mut sizes = copies.iter().map(|copy| copy.2).collect::<Vec<_>>();
    let mut attributes = MemcpyAttributes {
        // All sources were produced earlier on this same stream.
        src_access_order: 1,
        src_location: MemLocation {
            location_type: MEM_LOCATION_TYPE_DEVICE,
            id: device_ordinal,
        },
        dst_location: MemLocation {
            location_type: MEM_LOCATION_TYPE_DEVICE,
            id: device_ordinal,
        },
        flags: 0,
    };
    let mut attributes_index = 0;
    let mut failure_index = usize::MAX;
    let result = unsafe {
        (api.memcpy_batch_async)(
            destinations.as_mut_ptr(),
            sources.as_mut_ptr(),
            sizes.as_mut_ptr(),
            copies.len(),
            &mut attributes,
            &mut attributes_index,
            1,
            &mut failure_index,
            stream,
        )
    };
    if result.result().is_err() {
        // This is an optional optimization. A driver/API mismatch must not make
        // model execution fail; the caller will repeat the copies individually.
        DISABLED.store(true, Ordering::Relaxed);
        eprintln!(
            "cuMemcpyBatchAsync returned {result:?} at copy {failure_index}; disabling batched copies"
        );
    }
    Some(result)
}
