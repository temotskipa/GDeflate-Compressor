using System;
using System.IO;
using System.Runtime.InteropServices;

namespace GDeflateGUI
{
    public class GDeflateProcessor
    {
        // Helper method to allocate a device buffer and copy an array of pointers/sizes to it.
        private void CreateDevicePointerArray(IntPtr[] hostArray, out IntPtr devicePtr, out IntPtr hostPtr, CudaRuntimeApi.CudaMemcpyKind kind, IntPtr stream)
        {
            int sizeInBytes = hostArray.Length * Marshal.SizeOf<IntPtr>();
            hostPtr = Marshal.AllocHGlobal(sizeInBytes);
            Marshal.Copy(hostArray, 0, hostPtr, hostArray.Length);

            CheckCuda(CudaRuntimeApi.cudaMalloc(out devicePtr, (UIntPtr)sizeInBytes));
            CheckCuda(CudaRuntimeApi.cudaMemcpy(devicePtr, hostPtr, (UIntPtr)sizeInBytes, kind));
        }

        private void CreateDeviceSizeArray(UIntPtr[] hostArray, out IntPtr devicePtr, out IntPtr hostPtr, CudaRuntimeApi.CudaMemcpyKind kind, IntPtr stream)
        {
            // Marshal.Copy doesn't support UIntPtr, so we handle it as IntPtr (same size).
            var hostArrayAsIntPtr = Array.ConvertAll(hostArray, p => (IntPtr)p.ToUInt64());
            int sizeInBytes = hostArray.Length * Marshal.SizeOf<IntPtr>();

            hostPtr = Marshal.AllocHGlobal(sizeInBytes);
            Marshal.Copy(hostArrayAsIntPtr, 0, hostPtr, hostArray.Length);

            CheckCuda(CudaRuntimeApi.cudaMalloc(out devicePtr, (UIntPtr)sizeInBytes));
            CheckCuda(CudaRuntimeApi.cudaMemcpy(devicePtr, hostPtr, (UIntPtr)sizeInBytes, kind));
        }

        public void DecompressFile(string inputFile, string outputFile)
        {
            byte[] hostCompressedData = File.ReadAllBytes(inputFile);
            UIntPtr compressedSize = (UIntPtr)hostCompressedData.Length;

            IntPtr deviceCompressedPtr = IntPtr.Zero;
            IntPtr deviceDecompressedPtr = IntPtr.Zero;
            IntPtr deviceTempPtr = IntPtr.Zero;
            IntPtr deviceDecompressedSizePtr = IntPtr.Zero;
            IntPtr stream = IntPtr.Zero;

            IntPtr hostCompressedPtrs = IntPtr.Zero;
            IntPtr deviceCompressedPtrs = IntPtr.Zero;
            IntPtr hostCompressedSizes = IntPtr.Zero;
            IntPtr deviceCompressedSizes = IntPtr.Zero;
            IntPtr hostDecompressedPtrs = IntPtr.Zero;
            IntPtr deviceDecompressedPtrs = IntPtr.Zero;
            IntPtr hostDecompressedSizes = IntPtr.Zero;
            IntPtr deviceDecompressedSizes = IntPtr.Zero;

            try
            {
                CheckCuda(CudaRuntimeApi.cudaStreamCreate(out stream));
                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceCompressedPtr, compressedSize));
                CheckCuda(CudaRuntimeApi.cudaMemcpy(deviceCompressedPtr, Marshal.UnsafeAddrOfPinnedArrayElement(hostCompressedData, 0), compressedSize, CudaRuntimeApi.CudaMemcpyKind.HostToDevice));
                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceDecompressedSizePtr, (UIntPtr)Marshal.SizeOf<UIntPtr>()));

                CreateDevicePointerArray(new[] { deviceCompressedPtr }, out deviceCompressedPtrs, out hostCompressedPtrs, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);
                CreateDeviceSizeArray(new[] { compressedSize }, out deviceCompressedSizes, out hostCompressedSizes, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);

                CheckNvComp(NvCompApi.nvcompBatchedGdeflateGetDecompressSizeAsync(deviceCompressedPtrs, deviceCompressedSizes, deviceDecompressedSizePtr, (UIntPtr)1, stream));
                CheckCuda(CudaRuntimeApi.cudaStreamSynchronize(stream));

                var decompressedSizeArr = new UIntPtr[1];
                CheckCuda(CudaRuntimeApi.cudaMemcpy(Marshal.UnsafeAddrOfPinnedArrayElement(decompressedSizeArr, 0), deviceDecompressedSizePtr, (UIntPtr)Marshal.SizeOf<UIntPtr>(), CudaRuntimeApi.CudaMemcpyKind.DeviceToHost));
                UIntPtr decompressedSize = decompressedSizeArr[0];

                if (decompressedSize == UIntPtr.Zero) throw new InvalidDataException("Decompression size is zero. The file may be corrupt or not a valid GDeflate stream.");

                var decompressOpts = new NvCompApi.NvcompBatchedGdeflateDecompressOpts { backend = 0 };
                CheckNvComp(NvCompApi.nvcompBatchedGdeflateDecompressGetTempSizeAsync((UIntPtr)1, decompressedSize, decompressOpts, out UIntPtr tempSize, stream));
                CheckCuda(CudaRuntimeApi.cudaStreamSynchronize(stream));

                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceDecompressedPtr, decompressedSize));
                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceTempPtr, tempSize));

                CreateDevicePointerArray(new[] { deviceDecompressedPtr }, out deviceDecompressedPtrs, out hostDecompressedPtrs, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);
                CreateDeviceSizeArray(new[] { decompressedSize }, out deviceDecompressedSizes, out hostDecompressedSizes, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);

                CheckNvComp(NvCompApi.nvcompBatchedGdeflateDecompressAsync(deviceCompressedPtrs, deviceCompressedSizes, deviceDecompressedSizes, IntPtr.Zero, (UIntPtr)1, deviceTempPtr, tempSize, deviceDecompressedPtrs, decompressOpts, IntPtr.Zero, stream));
                CheckCuda(CudaRuntimeApi.cudaStreamSynchronize(stream));

                byte[] hostDecompressedData = new byte[(int)decompressedSize];
                CheckCuda(CudaRuntimeApi.cudaMemcpy(Marshal.UnsafeAddrOfPinnedArrayElement(hostDecompressedData, 0), deviceDecompressedPtr, decompressedSize, CudaRuntimeApi.CudaMemcpyKind.DeviceToHost));
                File.WriteAllBytes(outputFile, hostDecompressedData);
            }
            finally
            {
                if (hostCompressedPtrs != IntPtr.Zero) Marshal.FreeHGlobal(hostCompressedPtrs);
                if (hostCompressedSizes != IntPtr.Zero) Marshal.FreeHGlobal(hostCompressedSizes);
                if (hostDecompressedPtrs != IntPtr.Zero) Marshal.FreeHGlobal(hostDecompressedPtrs);
                if (hostDecompressedSizes != IntPtr.Zero) Marshal.FreeHGlobal(hostDecompressedSizes);

                if (deviceCompressedPtrs != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceCompressedPtrs);
                if (deviceCompressedSizes != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceCompressedSizes);
                if (deviceDecompressedPtrs != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceDecompressedPtrs);
                if (deviceDecompressedSizes != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceDecompressedSizes);

                if (deviceCompressedPtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceCompressedPtr);
                if (deviceDecompressedPtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceDecompressedPtr);
                if (deviceTempPtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceTempPtr);
                if (deviceDecompressedSizePtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceDecompressedSizePtr);
                if (stream != IntPtr.Zero) CudaRuntimeApi.cudaStreamDestroy(stream);
            }
        }

        public void CompressFile(string inputFile, string outputFile)
        {
            byte[] hostInputData = File.ReadAllBytes(inputFile);
            UIntPtr inputSize = (UIntPtr)hostInputData.Length;

            IntPtr deviceInputPtr = IntPtr.Zero;
            IntPtr deviceCompressedPtr = IntPtr.Zero;
            IntPtr deviceTempPtr = IntPtr.Zero;
            IntPtr deviceCompressedSizePtr = IntPtr.Zero;
            IntPtr stream = IntPtr.Zero;

            IntPtr hostInputPtrs = IntPtr.Zero;
            IntPtr deviceInputPtrs = IntPtr.Zero;
            IntPtr hostInputSizes = IntPtr.Zero;
            IntPtr deviceInputSizes = IntPtr.Zero;
            IntPtr hostCompressedPtrs = IntPtr.Zero;
            IntPtr deviceCompressedPtrs = IntPtr.Zero;

            try
            {
                CheckCuda(CudaRuntimeApi.cudaStreamCreate(out stream));
                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceInputPtr, inputSize));
                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceCompressedSizePtr, (UIntPtr)Marshal.SizeOf<UIntPtr>()));
                CheckCuda(CudaRuntimeApi.cudaMemcpy(deviceInputPtr, Marshal.UnsafeAddrOfPinnedArrayElement(hostInputData, 0), inputSize, CudaRuntimeApi.CudaMemcpyKind.HostToDevice));

                var compOpts = new NvCompApi.NvcompBatchedGdeflateCompressOpts { algorithm = 1 };
                CheckNvComp(NvCompApi.nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(inputSize, compOpts, out UIntPtr maxCompressedSize));
                CheckNvComp(NvCompApi.nvcompBatchedGdeflateCompressGetTempSizeAsync((UIntPtr)1, inputSize, compOpts, out UIntPtr tempSize, stream));
                CheckCuda(CudaRuntimeApi.cudaStreamSynchronize(stream));

                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceCompressedPtr, maxCompressedSize));
                CheckCuda(CudaRuntimeApi.cudaMalloc(out deviceTempPtr, tempSize));

                CreateDevicePointerArray(new[] { deviceInputPtr }, out deviceInputPtrs, out hostInputPtrs, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);
                CreateDeviceSizeArray(new[] { inputSize }, out deviceInputSizes, out hostInputSizes, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);
                CreateDevicePointerArray(new[] { deviceCompressedPtr }, out deviceCompressedPtrs, out hostCompressedPtrs, CudaRuntimeApi.CudaMemcpyKind.HostToDevice, stream);

                CheckNvComp(NvCompApi.nvcompBatchedGdeflateCompressAsync(deviceInputPtrs, deviceInputSizes, inputSize, (UIntPtr)1, deviceTempPtr, tempSize, deviceCompressedPtrs, deviceCompressedSizePtr, compOpts, IntPtr.Zero, stream));
                CheckCuda(CudaRuntimeApi.cudaStreamSynchronize(stream));

                var compressedSizeArr = new UIntPtr[1];
                CheckCuda(CudaRuntimeApi.cudaMemcpy(Marshal.UnsafeAddrOfPinnedArrayElement(compressedSizeArr, 0), deviceCompressedSizePtr, (UIntPtr)Marshal.SizeOf<UIntPtr>(), CudaRuntimeApi.CudaMemcpyKind.DeviceToHost));
                UIntPtr compressedSize = compressedSizeArr[0];

                byte[] hostOutputData = new byte[(int)compressedSize];
                CheckCuda(CudaRuntimeApi.cudaMemcpy(Marshal.UnsafeAddrOfPinnedArrayElement(hostOutputData, 0), deviceCompressedPtr, compressedSize, CudaRuntimeApi.CudaMemcpyKind.DeviceToHost));
                File.WriteAllBytes(outputFile, hostOutputData);
            }
            finally
            {
                if (hostInputPtrs != IntPtr.Zero) Marshal.FreeHGlobal(hostInputPtrs);
                if (hostInputSizes != IntPtr.Zero) Marshal.FreeHGlobal(hostInputSizes);
                if (hostCompressedPtrs != IntPtr.Zero) Marshal.FreeHGlobal(hostCompressedPtrs);

                if (deviceInputPtrs != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceInputPtrs);
                if (deviceInputSizes != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceInputSizes);
                if (deviceCompressedPtrs != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceCompressedPtrs);

                if (deviceInputPtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceInputPtr);
                if (deviceCompressedPtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceCompressedPtr);
                if (deviceTempPtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceTempPtr);
                if (deviceCompressedSizePtr != IntPtr.Zero) CudaRuntimeApi.cudaFree(deviceCompressedSizePtr);
                if (stream != IntPtr.Zero) CudaRuntimeApi.cudaStreamDestroy(stream);
            }
        }

        private void CheckCuda(int err)
        {
            if (err != 0)
            {
                IntPtr errStrPtr = CudaRuntimeApi.cudaGetErrorString(err);
                string errStr = Marshal.PtrToStringAnsi(errStrPtr);
                throw new InvalidOperationException($"CUDA Error: {errStr} (Code: {err})");
            }
        }

        private void CheckNvComp(NvCompApi.NvcompStatus status)
        {
            if (status != NvCompApi.NvcompStatus.nvcompSuccess)
            {
                throw new InvalidOperationException($"nvCOMP Error: {status}");
            }
        }
    }
}
