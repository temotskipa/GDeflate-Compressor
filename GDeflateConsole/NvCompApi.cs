using System;
using System.Runtime.InteropServices;

namespace GDeflateConsole
{
    internal static class NvCompApi
    {
        private static readonly IntPtr _nvcompHandle;
        private static readonly bool _isAvailable;

        #region Delegates
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate NvcompStatus nvcompBatchedGdeflateCompressGetTempSizeAsync_t(UIntPtr batchSize, UIntPtr maxUncompressedChunkBytes, NvcompBatchedGdeflateCompressOpts formatOpts, out UIntPtr tempBytes, IntPtr stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate NvcompStatus nvcompBatchedGdeflateCompressGetMaxOutputChunkSize_t(UIntPtr maxUncompressedChunkBytes, NvcompBatchedGdeflateCompressOpts formatOpts, out UIntPtr maxCompressedBytes);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate NvcompStatus nvcompBatchedGdeflateCompressAsync_t(IntPtr deviceUncompressedPtrs, IntPtr deviceUncompressedBytes, UIntPtr maxUncompressedChunkBytes, UIntPtr batchSize, IntPtr deviceTempPtr, UIntPtr tempBytes, IntPtr deviceCompressedPtrs, IntPtr deviceCompressedBytes, NvcompBatchedGdeflateCompressOpts formatOpts, IntPtr statusPtr, IntPtr stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate NvcompStatus nvcompBatchedGdeflateGetDecompressSizeAsync_t(IntPtr deviceCompressedPtrs, IntPtr deviceCompressedBytes, IntPtr deviceUncompressedBytes, UIntPtr batchSize, IntPtr stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate NvcompStatus nvcompBatchedGdeflateDecompressGetTempSizeAsync_t(UIntPtr batchSize, UIntPtr maxUncompressedChunkBytes, NvcompBatchedGdeflateDecompressOpts formatOpts, out UIntPtr tempBytes, IntPtr stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate NvcompStatus nvcompBatchedGdeflateDecompressAsync_t(IntPtr deviceCompressedPtrs, IntPtr deviceCompressedBytes, IntPtr deviceUncompressedBytes, IntPtr statusPtr, UIntPtr batchSize, IntPtr deviceTempPtr, UIntPtr tempBytes, IntPtr deviceUncompressedPtrs, NvcompBatchedGdeflateDecompressOpts formatOpts, IntPtr actualUncompressedBytes, IntPtr stream);
        #endregion

        #region Function Pointers
        private static readonly nvcompBatchedGdeflateCompressGetTempSizeAsync_t? _nvcompBatchedGdeflateCompressGetTempSizeAsync;
        private static readonly nvcompBatchedGdeflateCompressGetMaxOutputChunkSize_t? _nvcompBatchedGdeflateCompressGetMaxOutputChunkSize;
        private static readonly nvcompBatchedGdeflateCompressAsync_t? _nvcompBatchedGdeflateCompressAsync;
        private static readonly nvcompBatchedGdeflateGetDecompressSizeAsync_t? _nvcompBatchedGdeflateGetDecompressSizeAsync;
        private static readonly nvcompBatchedGdeflateDecompressGetTempSizeAsync_t? _nvcompBatchedGdeflateDecompressGetTempSizeAsync;
        private static readonly nvcompBatchedGdeflateDecompressAsync_t? _nvcompBatchedGdeflateDecompressAsync;
        #endregion

        static NvCompApi()
        {
            if (string.IsNullOrEmpty(NativeLibrary.NvcompDllPath))
            {
                _isAvailable = false;
                return;
            }

            _nvcompHandle = Kernel32.LoadLibrary(NativeLibrary.NvcompDllPath);
            if (_nvcompHandle == IntPtr.Zero)
            {
                _isAvailable = false;
                return;
            }

            try
            {
                _nvcompBatchedGdeflateCompressGetTempSizeAsync = Marshal.GetDelegateForFunctionPointer<nvcompBatchedGdeflateCompressGetTempSizeAsync_t>(Kernel32.GetProcAddress(_nvcompHandle, "nvcompBatchedGdeflateCompressGetTempSizeAsync"));
                _nvcompBatchedGdeflateCompressGetMaxOutputChunkSize = Marshal.GetDelegateForFunctionPointer<nvcompBatchedGdeflateCompressGetMaxOutputChunkSize_t>(Kernel32.GetProcAddress(_nvcompHandle, "nvcompBatchedGdeflateCompressGetMaxOutputChunkSize"));
                _nvcompBatchedGdeflateCompressAsync = Marshal.GetDelegateForFunctionPointer<nvcompBatchedGdeflateCompressAsync_t>(Kernel32.GetProcAddress(_nvcompHandle, "nvcompBatchedGdeflateCompressAsync"));
                _nvcompBatchedGdeflateGetDecompressSizeAsync = Marshal.GetDelegateForFunctionPointer<nvcompBatchedGdeflateGetDecompressSizeAsync_t>(Kernel32.GetProcAddress(_nvcompHandle, "nvcompBatchedGdeflateGetDecompressSizeAsync"));
                _nvcompBatchedGdeflateDecompressGetTempSizeAsync = Marshal.GetDelegateForFunctionPointer<nvcompBatchedGdeflateDecompressGetTempSizeAsync_t>(Kernel32.GetProcAddress(_nvcompHandle, "nvcompBatchedGdeflateDecompressGetTempSizeAsync"));
                _nvcompBatchedGdeflateDecompressAsync = Marshal.GetDelegateForFunctionPointer<nvcompBatchedGdeflateDecompressAsync_t>(Kernel32.GetProcAddress(_nvcompHandle, "nvcompBatchedGdeflateDecompressAsync"));
                _isAvailable = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load nvCOMP functions: {ex.Message}");
                Kernel32.FreeLibrary(_nvcompHandle);
                _isAvailable = false;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NvcompBatchedGdeflateCompressOpts { public int algorithm; [MarshalAs(UnmanagedType.ByValArray, SizeConst = 60)] public byte[] reserved; }
        [StructLayout(LayoutKind.Sequential)]
        internal struct NvcompBatchedGdeflateDecompressOpts { public int backend; [MarshalAs(UnmanagedType.ByValArray, SizeConst = 60)] public byte[] reserved; }

        internal enum NvcompStatus { nvcompSuccess = 0, nvcompErrorInvalidValue = 1, nvcompErrorNotSupported = 2, nvcompErrorCannotDecompress = 3, nvcompErrorBadChecksum = 4, nvcompErrorCannotVerifyChecksums = 5, nvcompErrorOutputBufferTooSmall = 6, nvcompErrorWrongHeaderLength = 7, nvcompErrorAlignment = 8, nvcompErrorChunkSizeTooLarge = 9, nvcompErrorCannotCompress = 10, nvcompErrorWrongInputLength = 11, nvcompErrorCudaError = 12, nvcompErrorInternal = 13 }

        internal static NvcompStatus nvcompBatchedGdeflateCompressGetTempSizeAsync(UIntPtr batchSize, UIntPtr maxUncompressedChunkBytes, NvcompBatchedGdeflateCompressOpts formatOpts, out UIntPtr tempBytes, IntPtr stream)
        {
            if (_nvcompBatchedGdeflateCompressGetTempSizeAsync != null) return _nvcompBatchedGdeflateCompressGetTempSizeAsync(batchSize, maxUncompressedChunkBytes, formatOpts, out tempBytes, stream);
            tempBytes = UIntPtr.Zero;
            return NvcompStatus.nvcompErrorInternal;
        }
        internal static NvcompStatus nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(UIntPtr maxUncompressedChunkBytes, NvcompBatchedGdeflateCompressOpts formatOpts, out UIntPtr maxCompressedBytes)
        {
            if (_nvcompBatchedGdeflateCompressGetMaxOutputChunkSize != null) return _nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(maxUncompressedChunkBytes, formatOpts, out maxCompressedBytes);
            maxCompressedBytes = UIntPtr.Zero;
            return NvcompStatus.nvcompErrorInternal;
        }
        internal static NvcompStatus nvcompBatchedGdeflateCompressAsync(IntPtr deviceUncompressedPtrs, IntPtr deviceUncompressedBytes, UIntPtr maxUncompressedChunkBytes, UIntPtr batchSize, IntPtr deviceTempPtr, UIntPtr tempBytes, IntPtr deviceCompressedPtrs, IntPtr deviceCompressedBytes, NvcompBatchedGdeflateCompressOpts formatOpts, IntPtr statusPtr, IntPtr stream) => _nvcompBatchedGdeflateCompressAsync?.Invoke(deviceUncompressedPtrs, deviceUncompressedBytes, maxUncompressedChunkBytes, batchSize, deviceTempPtr, tempBytes, deviceCompressedPtrs, deviceCompressedBytes, formatOpts, statusPtr, stream) ?? NvcompStatus.nvcompErrorInternal;
        internal static NvcompStatus nvcompBatchedGdeflateGetDecompressSizeAsync(IntPtr deviceCompressedPtrs, IntPtr deviceCompressedBytes, IntPtr deviceUncompressedBytes, UIntPtr batchSize, IntPtr stream) => _nvcompBatchedGdeflateGetDecompressSizeAsync?.Invoke(deviceCompressedPtrs, deviceCompressedBytes, deviceUncompressedBytes, batchSize, stream) ?? NvcompStatus.nvcompErrorInternal;
        internal static NvcompStatus nvcompBatchedGdeflateDecompressGetTempSizeAsync(UIntPtr batchSize, UIntPtr maxUncompressedChunkBytes, NvcompBatchedGdeflateDecompressOpts formatOpts, out UIntPtr tempBytes, IntPtr stream)
        {
            if (_nvcompBatchedGdeflateDecompressGetTempSizeAsync != null) return _nvcompBatchedGdeflateDecompressGetTempSizeAsync(batchSize, maxUncompressedChunkBytes, formatOpts, out tempBytes, stream);
            tempBytes = UIntPtr.Zero;
            return NvcompStatus.nvcompErrorInternal;
        }
        internal static NvcompStatus nvcompBatchedGdeflateDecompressAsync(IntPtr deviceCompressedPtrs, IntPtr deviceCompressedBytes, IntPtr deviceUncompressedBytes, IntPtr statusPtr, UIntPtr batchSize, IntPtr deviceTempPtr, UIntPtr tempBytes, IntPtr deviceUncompressedPtrs, NvcompBatchedGdeflateDecompressOpts formatOpts, IntPtr actualUncompressedBytes, IntPtr stream) => _nvcompBatchedGdeflateDecompressAsync?.Invoke(deviceCompressedPtrs, deviceCompressedBytes, deviceUncompressedBytes, statusPtr, batchSize, deviceTempPtr, tempBytes, deviceUncompressedPtrs, formatOpts, actualUncompressedBytes, stream) ?? NvcompStatus.nvcompErrorInternal;

        internal static bool IsNvCompAvailable() => _isAvailable;
    }
}