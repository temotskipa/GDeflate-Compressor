using System;
using System.Runtime.InteropServices;

namespace GDeflateConsole
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
            UIntPtr batchSize,
            UIntPtr maxUncompressedChunkBytes,
            NvcompBatchedGdeflateCompressOpts formatOpts,
            out UIntPtr tempBytes,
            IntPtr stream);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateCompressGetMaxOutputChunkSize")]
        internal static extern NvcompStatus nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
            UIntPtr maxUncompressedChunkBytes,
            NvcompBatchedGdeflateCompressOpts formatOpts,
            out UIntPtr maxCompressedBytes);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateCompressAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateCompressAsync(
            IntPtr deviceUncompressedPtrs,
            IntPtr deviceUncompressedBytes,
            UIntPtr maxUncompressedChunkBytes,
            UIntPtr batchSize,
            IntPtr deviceTempPtr,
            UIntPtr tempBytes,
            IntPtr deviceCompressedPtrs,
            IntPtr deviceCompressedBytes,
            NvcompBatchedGdeflateCompressOpts formatOpts,
            IntPtr statusPtr,
            IntPtr stream);

        // --- Decompression Functions ---

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateGetDecompressSizeAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateGetDecompressSizeAsync(
            IntPtr deviceCompressedPtrs,
            IntPtr deviceCompressedBytes,
            IntPtr deviceUncompressedBytes,
            UIntPtr batchSize,
            IntPtr stream);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateDecompressGetTempSizeAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateDecompressGetTempSizeAsync(
            UIntPtr batchSize,
            UIntPtr maxUncompressedChunkBytes,
            NvcompBatchedGdeflateDecompressOpts formatOpts,
            out UIntPtr tempBytes,
            IntPtr stream);

        [DllImport(NvCompDll, EntryPoint = "nvcompBatchedGdeflateDecompressAsync")]
        internal static extern NvcompStatus nvcompBatchedGdeflateDecompressAsync(
            IntPtr deviceCompressedPtrs,
            IntPtr deviceCompressedBytes,
            IntPtr deviceUncompressedBytes,
            IntPtr statusPtr,
            UIntPtr batchSize,
            IntPtr deviceTempPtr,
            UIntPtr tempBytes,
            IntPtr deviceUncompressedPtrs,
            NvcompBatchedGdeflateDecompressOpts formatOpts,
            IntPtr actualUncompressedBytes,
            IntPtr stream);

        // Helper method to check if nvCOMP is available
        internal static bool IsNvCompAvailable()
        {
            try
            {
                // Try to call a simple function to test if the library is available
                var opts = new NvcompBatchedGdeflateCompressOpts { algorithm = 1 };
                var result = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize((UIntPtr)1024, opts, out UIntPtr maxSize);
                return true; // If we get here without exception, nvCOMP is available
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
            catch
            {
                return false;
            }
        }
    }
}