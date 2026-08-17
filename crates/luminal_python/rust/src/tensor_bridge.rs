//! Minimal PyTorch tensor interop for native invocation.
//!
//! PyTorch 2.13+ exposes the DLPack C exchange API on `torch.Tensor`. We cache
//! that function table and use its non-owning tensor view on the hot path.
//! Older PyTorch versions fall back to supported Python-visible properties.

use std::{ffi::c_void, ptr};

use pyo3::{
    exceptions::PyAttributeError,
    ffi,
    prelude::*,
    types::{PyByteArray, PyDict, PyModule},
};

use crate::torch_dtype::TorchDType;

const DLPACK_CAPSULE_NAME: &[u8] = b"dlpack_exchange_api\0";
const DL_DEVICE_CPU: i32 = 1;
const DL_DEVICE_CUDA: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DlTensor {
    data: *mut c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DlPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DlPackExchangeApiHeader {
    version: DlPackVersion,
    previous: *mut DlPackExchangeApiHeader,
}

type DlPackFn = Option<unsafe extern "C" fn()>;
type DlTensorFromPyObject = Option<unsafe extern "C" fn(*mut c_void, *mut DlTensor) -> i32>;

#[repr(C)]
struct DlPackExchangeApi {
    header: DlPackExchangeApiHeader,
    managed_tensor_allocator: DlPackFn,
    managed_tensor_from_py_object_no_sync: DlPackFn,
    managed_tensor_to_py_object_no_sync: DlPackFn,
    dltensor_from_py_object_no_sync: DlTensorFromPyObject,
    current_work_stream: DlPackFn,
}

/// Metadata needed by Luminal to validate and bind one PyTorch tensor.
pub(crate) struct TensorObservation {
    pub dtype_code: u32,
    pub device_type: i32,
    pub device_id: i32,
    pub is_contiguous: bool,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub element_size: usize,
    pub data_ptr: u64,
}

impl TensorObservation {
    pub fn is_cuda(&self) -> bool {
        self.device_type == DL_DEVICE_CUDA
    }

    pub fn is_cpu(&self) -> bool {
        self.device_type == DL_DEVICE_CPU
    }

    pub fn n_bytes(&self) -> usize {
        self.numel * self.element_size
    }
}

/// Cached Python constructors plus the optional native tensor-observation API.
pub(crate) struct TorchApi {
    empty: Py<PyAny>,
    tensor: Py<PyAny>,
    frombuffer: Py<PyAny>,
    cpu_device: Py<PyAny>,
    dtypes: Vec<(u32, Py<PyAny>)>,
    dlpack_exchange_api: Option<usize>,
}

impl TorchApi {
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let torch = PyModule::import(py, "torch")?;
        let dtypes = supported_dtypes()
            .into_iter()
            .map(|dtype| {
                torch
                    .getattr(dtype_attribute(dtype.code())?)
                    .map(|object| (dtype.code(), object.unbind()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let empty = torch.getattr("empty")?.unbind();
        let tensor = torch.getattr("tensor")?.unbind();
        let frombuffer = torch.getattr("frombuffer")?.unbind();
        let cpu_device = torch.getattr("device")?.call1(("cpu",))?.unbind();
        // The override is useful for compatibility tests against the Python
        // fallback without requiring an older PyTorch installation.
        let dlpack_exchange_api = if std::env::var_os("LUMINAL_DISABLE_DLPACK_C_EXCHANGE").is_some()
        {
            None
        } else {
            native_exchange_api(&torch)?
        };
        Ok(Self {
            empty,
            tensor,
            frombuffer,
            cpu_device,
            dtypes,
            dlpack_exchange_api,
        })
    }

    pub fn observe(
        &self,
        py: Python<'_>,
        tensor: &Bound<'_, PyAny>,
    ) -> PyResult<TensorObservation> {
        if let Some(api) = self.dlpack_exchange_api
            && let Some(observation) = unsafe { observe_native(py, tensor, api)? }
        {
            return Ok(observation);
        }
        self.observe_python(tensor)
    }

    fn observe_python(&self, tensor: &Bound<'_, PyAny>) -> PyResult<TensorObservation> {
        let dtype = tensor.getattr("dtype")?;
        let device = tensor.getattr("device")?;
        let device_type = device.getattr("type")?.extract::<String>()?;
        Ok(TensorObservation {
            dtype_code: self.dtype_code(&dtype)?,
            device_type: match device_type.as_str() {
                "cpu" => DL_DEVICE_CPU,
                "cuda" => DL_DEVICE_CUDA,
                _ => -1,
            },
            device_id: device
                .getattr("index")?
                .extract::<Option<i32>>()?
                .unwrap_or(0),
            is_contiguous: tensor.call_method0("is_contiguous")?.extract()?,
            shape: tensor.getattr("shape")?.extract()?,
            numel: tensor.call_method0("numel")?.extract()?,
            element_size: tensor.call_method0("element_size")?.extract()?,
            data_ptr: tensor.call_method0("data_ptr")?.extract()?,
        })
    }

    pub fn make_contiguous<'py>(&self, tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tensor.call_method0("detach")?.call_method0("contiguous")
    }

    pub fn make_cpu_contiguous<'py>(
        &self,
        tensor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tensor
            .call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("contiguous")
    }

    pub fn device<'py>(&self, tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tensor.getattr("device")
    }

    pub fn cpu_device<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.cpu_device.bind(py).clone()
    }

    pub fn dtype<'py>(&self, py: Python<'py>, code: u32) -> PyResult<Bound<'py, PyAny>> {
        self.dtypes
            .iter()
            .find_map(|(candidate, dtype)| (*candidate == code).then(|| dtype.bind(py).clone()))
            .ok_or_else(|| unsupported_dtype(code))
    }

    pub fn empty<'py>(
        &self,
        py: Python<'py>,
        shape: &[usize],
        dtype_code: u32,
        device: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", self.dtype(py, dtype_code)?)?;
        kwargs.set_item("device", device)?;
        self.empty.bind(py).call((shape.to_vec(),), Some(&kwargs))
    }

    pub fn tensor_from_values<'py, T>(
        &self,
        py: Python<'py>,
        values: Vec<T>,
        dtype_code: u32,
    ) -> PyResult<Bound<'py, PyAny>>
    where
        T: for<'a> IntoPyObject<'a> + Send,
        for<'a> <T as IntoPyObject<'a>>::Error: Into<PyErr>,
    {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", self.dtype(py, dtype_code)?)?;
        self.tensor.bind(py).call((values,), Some(&kwargs))
    }

    pub fn tensor_from_bytes<'py>(
        &self,
        py: Python<'py>,
        bytes: &[u8],
        dtype_code: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let buffer = PyByteArray::new(py, bytes);
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", self.dtype(py, dtype_code)?)?;
        self.frombuffer.bind(py).call((buffer,), Some(&kwargs))
    }

    pub fn reshape_to_device<'py>(
        &self,
        tensor: Bound<'py, PyAny>,
        shape: &[usize],
        device: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tensor
            .call_method1("reshape", (shape.to_vec(),))?
            .call_method1("to", (device,))
    }

    fn dtype_code(&self, dtype: &Bound<'_, PyAny>) -> PyResult<u32> {
        self.dtypes
            .iter()
            .find_map(|(code, candidate)| candidate.bind(dtype.py()).is(dtype).then_some(*code))
            .ok_or_else(|| {
                pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "PyTorch dtype {} is not supported by Luminal",
                    dtype.str().map_or_else(
                        |_| "<unprintable>".to_owned(),
                        |value| value
                            .extract()
                            .unwrap_or_else(|_| "<unprintable>".to_owned())
                    )
                ))
            })
    }
}

fn native_exchange_api(torch: &Bound<'_, PyModule>) -> PyResult<Option<usize>> {
    let tensor_type = torch.getattr("Tensor")?;
    let capsule = match tensor_type.getattr("__dlpack_c_exchange_api__") {
        Ok(capsule) => capsule,
        Err(error) if error.is_instance_of::<PyAttributeError>(torch.py()) => return Ok(None),
        Err(error) => return Err(error),
    };
    let pointer = unsafe {
        ffi::PyCapsule_GetPointer(
            capsule.as_ptr(),
            DLPACK_CAPSULE_NAME.as_ptr().cast::<std::ffi::c_char>(),
        )
    };
    if pointer.is_null() {
        return Err(PyErr::fetch(torch.py()));
    }
    let api = unsafe { &*pointer.cast::<DlPackExchangeApi>() };
    if api.header.version.major != 1 || api.dltensor_from_py_object_no_sync.is_none() {
        return Ok(None);
    }
    Ok(Some(pointer as usize))
}

unsafe fn observe_native(
    py: Python<'_>,
    tensor: &Bound<'_, PyAny>,
    api_pointer: usize,
) -> PyResult<Option<TensorObservation>> {
    let api = unsafe { &*(api_pointer as *const DlPackExchangeApi) };
    let Some(export) = api.dltensor_from_py_object_no_sync else {
        return Ok(None);
    };
    let mut view = DlTensor {
        data: ptr::null_mut(),
        device: DlDevice {
            device_type: 0,
            device_id: 0,
        },
        ndim: 0,
        dtype: DlDataType {
            code: 0,
            bits: 0,
            lanes: 0,
        },
        shape: ptr::null_mut(),
        strides: ptr::null_mut(),
        byte_offset: 0,
    };
    if unsafe { export(tensor.as_ptr().cast(), &mut view) } != 0 {
        let _ = PyErr::fetch(py);
        return Ok(None);
    }
    if view.ndim < 0 || (view.ndim > 0 && view.shape.is_null()) {
        return Ok(None);
    }
    let rank = view.ndim as usize;
    let raw_shape = unsafe { std::slice::from_raw_parts(view.shape, rank) };
    let shape = raw_shape
        .iter()
        .map(|&dimension| usize::try_from(dimension).ok())
        .collect::<Option<Vec<_>>>();
    let Some(shape) = shape else {
        return Ok(None);
    };
    let dtype_code = match dlpack_dtype_code(view.dtype) {
        Some(code) => code,
        None => return Ok(None),
    };
    let element_bits = usize::from(view.dtype.bits) * usize::from(view.dtype.lanes);
    if element_bits == 0 || !element_bits.is_multiple_of(8) {
        return Ok(None);
    }
    let element_size = element_bits / 8;
    let numel = shape.iter().product();
    let is_contiguous = dlpack_is_contiguous(&shape, view.strides);
    let address = (view.data as usize).checked_add(view.byte_offset as usize);
    let Some(address) = address else {
        return Ok(None);
    };
    Ok(Some(TensorObservation {
        dtype_code,
        device_type: view.device.device_type,
        device_id: view.device.device_id,
        is_contiguous,
        shape,
        numel,
        element_size,
        data_ptr: address as u64,
    }))
}

fn dlpack_is_contiguous(shape: &[usize], strides: *mut i64) -> bool {
    if strides.is_null() || shape.contains(&0) {
        return true;
    }
    let strides = unsafe { std::slice::from_raw_parts(strides, shape.len()) };
    let mut expected = 1usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        if dimension > 1 && usize::try_from(stride).ok() != Some(expected) {
            return false;
        }
        let Some(next) = expected.checked_mul(dimension) else {
            return false;
        };
        expected = next;
    }
    true
}

fn dlpack_dtype_code(dtype: DlDataType) -> Option<u32> {
    if dtype.lanes != 1 {
        return None;
    }
    Some(
        match (dtype.code, dtype.bits) {
            (1, 8) => TorchDType::Byte,
            (0, 8) => TorchDType::Char,
            (0, 16) => TorchDType::Short,
            (0, 32) => TorchDType::Int,
            (0, 64) => TorchDType::Long,
            (2, 16) => TorchDType::Half,
            (2, 32) => TorchDType::Float,
            (2, 64) => TorchDType::Double,
            (6, 8) => TorchDType::Bool,
            (4, 16) => TorchDType::BFloat16,
            _ => return None,
        }
        .code(),
    )
}

fn supported_dtypes() -> [TorchDType; 10] {
    [
        TorchDType::Byte,
        TorchDType::Char,
        TorchDType::Short,
        TorchDType::Int,
        TorchDType::Long,
        TorchDType::Half,
        TorchDType::Float,
        TorchDType::Double,
        TorchDType::Bool,
        TorchDType::BFloat16,
    ]
}

fn dtype_attribute(code: u32) -> PyResult<&'static str> {
    match TorchDType::from_code(code) {
        Ok(TorchDType::Byte) => Ok("uint8"),
        Ok(TorchDType::Char) => Ok("int8"),
        Ok(TorchDType::Short) => Ok("int16"),
        Ok(TorchDType::Int) => Ok("int32"),
        Ok(TorchDType::Long) => Ok("int64"),
        Ok(TorchDType::Half) => Ok("float16"),
        Ok(TorchDType::Float) => Ok("float32"),
        Ok(TorchDType::Double) => Ok("float64"),
        Ok(TorchDType::Bool) => Ok("bool"),
        Ok(TorchDType::BFloat16) => Ok("bfloat16"),
        _ => Err(unsupported_dtype(code)),
    }
}

fn unsupported_dtype(code: u32) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "PT2 dtype code {code} is not supported by Luminal's tensor bridge"
    ))
}

pub(crate) fn is_zero_copy_output_dtype(code: u32) -> bool {
    matches!(
        TorchDType::from_code(code),
        Ok(TorchDType::Float | TorchDType::Half | TorchDType::BFloat16)
    )
}
