using System;
using System.Runtime.InteropServices;

namespace GDeflateConsole
{
    internal static class CudaRuntimeApi
    {
        // The name of the CUDA Runtime DLL. This may be version-specific
        // (e.g., "cudart64_110.dll", "cudart64_12.dll").
        // It must be in the system's PATH or alongside the executable.
        private const string CudaRuntimeDll = "cudart64_12.dll";

        internal enum CudaMemcpyKind
        {
            HostToHost = 0,
            HostToDevice = 1,
            DeviceToHost = 2,
            DeviceToDevice = 3,
            Default = 4
        }

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaMalloc(out IntPtr devPtr, UIntPtr size);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaMemcpy(IntPtr dst, IntPtr src, UIntPtr count, CudaMemcpyKind kind);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaFree(IntPtr devPtr);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaStreamCreate(out IntPtr pStream);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaStreamDestroy(IntPtr stream);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaStreamSynchronize(IntPtr stream);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern IntPtr cudaGetErrorString(int error);

        [DllImport(CudaRuntimeDll, SetLastError = true)]
        internal static extern int cudaGetDeviceCount(out int count);

        // Helper method to check if CUDA is available
        internal static bool IsCudaAvailable()
        {
            try
            {
                int deviceCount;
                int result = cudaGetDeviceCount(out deviceCount);
                return result == 0 && deviceCount > 0;
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