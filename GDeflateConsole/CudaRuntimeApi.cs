using System;
using System.Runtime.InteropServices;

namespace GDeflateConsole
{
    internal static class CudaRuntimeApi
    {
        private static readonly IntPtr _cudaRuntimeHandle;
        private static readonly bool _isAvailable;

        #region Delegates
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaMalloc_t(out IntPtr devPtr, UIntPtr size);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaMemcpy_t(IntPtr dst, IntPtr src, UIntPtr count, CudaMemcpyKind kind);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaFree_t(IntPtr devPtr);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaStreamCreate_t(out IntPtr pStream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaStreamDestroy_t(IntPtr stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaStreamSynchronize_t(IntPtr stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr cudaGetErrorString_t(int error);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaGetDeviceCount_t(out int count);
        #endregion

        #region Function Pointers
        private static readonly cudaMalloc_t? _cudaMalloc;
        private static readonly cudaMemcpy_t? _cudaMemcpy;
        private static readonly cudaFree_t? _cudaFree;
        private static readonly cudaStreamCreate_t? _cudaStreamCreate;
        private static readonly cudaStreamDestroy_t? _cudaStreamDestroy;
        private static readonly cudaStreamSynchronize_t? _cudaStreamSynchronize;
        private static readonly cudaGetErrorString_t? _cudaGetErrorString;
        private static readonly cudaGetDeviceCount_t? _cudaGetDeviceCount;
        #endregion

        static CudaRuntimeApi()
        {
            if (string.IsNullOrEmpty(NativeLibrary.CudartDllPath))
            {
                _isAvailable = false;
                return;
            }

            _cudaRuntimeHandle = Kernel32.LoadLibrary(NativeLibrary.CudartDllPath);
            if (_cudaRuntimeHandle == IntPtr.Zero)
            {
                _isAvailable = false;
                return;
            }

            try
            {
                _cudaMalloc = Marshal.GetDelegateForFunctionPointer<cudaMalloc_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaMalloc"));
                _cudaMemcpy = Marshal.GetDelegateForFunctionPointer<cudaMemcpy_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaMemcpy"));
                _cudaFree = Marshal.GetDelegateForFunctionPointer<cudaFree_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaFree"));
                _cudaStreamCreate = Marshal.GetDelegateForFunctionPointer<cudaStreamCreate_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaStreamCreate"));
                _cudaStreamDestroy = Marshal.GetDelegateForFunctionPointer<cudaStreamDestroy_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaStreamDestroy"));
                _cudaStreamSynchronize = Marshal.GetDelegateForFunctionPointer<cudaStreamSynchronize_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaStreamSynchronize"));
                _cudaGetErrorString = Marshal.GetDelegateForFunctionPointer<cudaGetErrorString_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaGetErrorString"));
                _cudaGetDeviceCount = Marshal.GetDelegateForFunctionPointer<cudaGetDeviceCount_t>(Kernel32.GetProcAddress(_cudaRuntimeHandle, "cudaGetDeviceCount"));
                _isAvailable = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load CUDA runtime functions: {ex.Message}");
                Kernel32.FreeLibrary(_cudaRuntimeHandle);
                _isAvailable = false;
            }
        }

        internal enum CudaMemcpyKind
        {
            HostToHost = 0,
            HostToDevice = 1,
            DeviceToHost = 2,
            DeviceToDevice = 3,
            Default = 4
        }

        internal static int cudaMalloc(out IntPtr devPtr, UIntPtr size)
        {
            if (_cudaMalloc != null) return _cudaMalloc(out devPtr, size);
            devPtr = IntPtr.Zero;
            return -1;
        }
        internal static int cudaMemcpy(IntPtr dst, IntPtr src, UIntPtr count, CudaMemcpyKind kind) => _cudaMemcpy?.Invoke(dst, src, count, kind) ?? -1;
        internal static int cudaFree(IntPtr devPtr) => _cudaFree?.Invoke(devPtr) ?? -1;
        internal static int cudaStreamCreate(out IntPtr pStream)
        {
            if (_cudaStreamCreate != null) return _cudaStreamCreate(out pStream);
            pStream = IntPtr.Zero;
            return -1;
        }
        internal static int cudaStreamDestroy(IntPtr stream) => _cudaStreamDestroy?.Invoke(stream) ?? -1;
        internal static int cudaStreamSynchronize(IntPtr stream) => _cudaStreamSynchronize?.Invoke(stream) ?? -1;
        internal static IntPtr cudaGetErrorString(int error) => _cudaGetErrorString?.Invoke(error) ?? IntPtr.Zero;
        internal static int cudaGetDeviceCount(out int count)
        {
            if (_cudaGetDeviceCount != null)
                return _cudaGetDeviceCount(out count);
            count = 0;
            return -1;
        }

        internal static bool IsCudaAvailable()
        {
            if (!_isAvailable)
                return false;

            try
            {
                int deviceCount;
                int result = cudaGetDeviceCount(out deviceCount);
                return result == 0 && deviceCount > 0;
            }
            catch
            {
                return false;
            }
        }
    }
}