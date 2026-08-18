//! Opt-in NVTX ranges for Nsight Systems invocation profiling.

pub(crate) struct Range {
    #[cfg(feature = "cuda")]
    active: bool,
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::{
        ffi::{CStr, c_char},
        sync::OnceLock,
    };

    type RangePush = unsafe extern "C" fn(*const c_char) -> i32;
    type RangePop = unsafe extern "C" fn() -> i32;

    pub(super) struct Api {
        _library: libloading::Library,
        push: RangePush,
        pop: RangePop,
    }

    static API: OnceLock<Option<Api>> = OnceLock::new();

    pub(super) fn api() -> Option<&'static Api> {
        API.get_or_init(|| {
            std::env::var_os("LUMINAL_NVTX")?;
            let library = ["libnvToolsExt.so.1", "libnvToolsExt.so"]
                .into_iter()
                .find_map(|name| unsafe { libloading::Library::new(name).ok() });
            let Some(library) = library else {
                eprintln!("LUMINAL_NVTX is set, but libnvToolsExt could not be loaded");
                return None;
            };
            unsafe {
                let push = *library.get::<RangePush>(b"nvtxRangePushA\0").ok()?;
                let pop = *library.get::<RangePop>(b"nvtxRangePop\0").ok()?;
                Some(Api {
                    _library: library,
                    push,
                    pop,
                })
            }
        })
        .as_ref()
    }

    pub(super) fn push(api: &Api, name: &CStr) {
        unsafe { (api.push)(name.as_ptr()) };
    }

    pub(super) fn pop(api: &Api) {
        unsafe { (api.pop)() };
    }
}

pub(crate) fn range(name: &'static std::ffi::CStr) -> Range {
    #[cfg(feature = "cuda")]
    {
        let active = cuda::api().is_some_and(|api| {
            cuda::push(api, name);
            true
        });
        Range { active }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = name;
        Range {}
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if self.active
            && let Some(api) = cuda::api()
        {
            cuda::pop(api);
        }
    }
}
