using System;
using System.Runtime.InteropServices;

namespace GDeflateGUI
{
    internal static class NvCompApi
    {
        // The name of the nvCOMP DLL.
        // It must be in the system's PATH or alongside the executable.
        private const string NvCompDll = "nvcomp.dll";

        [StructLayout(LayoutKind.Sequential)]
        internal struct NvcompBatchedGdeflateCompressOpts
        {
            public int algorithm;
            // The C struct has a `char reserved[60]` member.
            // We can represent this with a byte array and ensure the struct size is correct.
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 60)]
            public byte[] reserved;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NvcompBatchedGdeflateDecompressOpts
        {
            public int backend; // This corresponds to nvcompDecompressBackend_t enum, but int is safer for P/Invoke
             [MarshalAs(UnmanagedType.ByValArray, SizeConst = 60)]
            public byte[] reserved;
        }

        internal enum NvcompStatus
        {
            nvcompSuccess = 0,
            nvcompErrorInvalidValue = 1,
            nvcompErrorNotSupported = 2,
            nvcompErrorCannotDecompress = 3,
            nvcompErrorBadChecksum = 4,
            nvcompErrorCannotVerifyChecksums = 5,
            nvcompErrorOutputBufferTooSmall = 6,
            nvcompErrorWrongHeaderLength = 7,
            nvcompErrorAlignment = 8,
            nvcompErrorChunkSizeTooLarge = 9,
            nvcompErrorCannotCompress = 10,
            nvcompErrorWrongInputLength = 11,
            nvcompErrorCudaError = 12,
            nvcompErrorInternal = 13
        }

        // --- Compression Functions ---

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateCompressGetTempSizeAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateCompressGetTempSizeAsync(
            UIntPtr numChunks,
            UIntPtr maxUncompressedChunkBytes,
            NvcompBatchedGdeflateCompressOpts opts,
            out UIntPtr tempBytes,
            IntPtr stream);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateCompressGetMaxOutputChunkSize")]
        internal static extern NvcompStatus nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
            UIntPtr maxUncompressedChunkBytes,
            NvcompBatchedGdeflateCompressOpts opts,
            out UIntPtr maxCompressedChunkBytes);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateCompressAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateCompressAsync(
            IntPtr deviceUncompressedChunkPtrs, // const void* const*
            IntPtr deviceUncompressedChunkBytes, // const size_t*
            UIntPtr maxUncompressedChunkBytes,
            UIntPtr numChunks,
            IntPtr deviceTempPtr,
            UIntPtr tempBytes,
            IntPtr deviceCompressedChunkPtrs, // void* const*
            IntPtr deviceCompressedChunkBytes, // size_t*
            NvcompBatchedGdeflateCompressOpts opts,
            IntPtr deviceStatuses, // nvcompStatus_t*
            IntPtr stream);

        // --- Decompression Functions ---

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateGetDecompressSizeAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateGetDecompressSizeAsync(
            IntPtr deviceCompressedChunkPtrs, // const void* const*
            IntPtr deviceCompressedChunkBytes, // const size_t*
            IntPtr deviceUncompressedChunkBytes, // size_t*
            UIntPtr numChunks,
            IntPtr stream);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateDecompressGetTempSizeAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateDecompressGetTempSizeAsync(
            UIntPtr numChunks,
            UIntPtr maxUncompressedChunkBytes,
            NvcompBatchedGdeflateDecompressOpts opts,
            out UIntPtr tempBytes,
            IntPtr stream);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateDecompressAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateDecompressAsync(
            IntPtr deviceCompressedChunkPtrs, // const void* const*
            IntPtr deviceCompressedChunkBytes, // const size_t*
            IntPtr deviceUncompressedBufferBytes, // const size_t*
            IntPtr deviceActualUncompressedChunkBytes, // size_t*
            UIntPtr numChunks,
            IntPtr deviceTempPtr,
            UIntPtr tempBytes,
            IntPtr deviceUncompressedChunkPtrs, // void* const*
            NvcompBatchedGdeflateDecompressOpts decompressOpts,
            IntPtr deviceStatuses, // nvcompStatus_t*
            IntPtr stream);
    }
}
