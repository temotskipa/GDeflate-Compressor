using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace GDeflateConsole
{
    internal static class NativeLibrary
    {
        private static string? _cudaToolkitPath;
        private static string? _cudartDllPath;
        private static string? _nvcompDllPath;

        static NativeLibrary()
        {
            try
            {
                _cudaToolkitPath = FindCudaToolkitPath();
                if (_cudaToolkitPath != null)
                {
                    _cudartDllPath = FindCudart();
                    _nvcompDllPath = FindNvcomp();
                }
            }
            catch (Exception ex)
            {
                // Log initialization errors if necessary
                Console.WriteLine($"Error during native library initialization: {ex.Message}");
            }
        }

        private static string? FindCudaToolkitPath()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
                string nvidiaGpuComputingToolkit = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
                if (Directory.Exists(nvidiaGpuComputingToolkit))
                {
                    var versions = Directory.GetDirectories(nvidiaGpuComputingToolkit, "v*.*")
                        .Select(path => new { Path = path, Version = GetVersionFromPath(path) })
                        .OrderByDescending(x => x.Version)
                        .ToList();

                    return versions.FirstOrDefault()?.Path;
                }
            }
            else
            {
                // Linux/Mac default paths
                string[] commonPaths = { "/usr/local/cuda", "/opt/cuda" };
                foreach (var path in commonPaths)
                {
                    if (Directory.Exists(path))
                    {
                        return path;
                    }
                }
            }
            return null;
        }

        private static Version? GetVersionFromPath(string path)
        {
            var dirName = new DirectoryInfo(path).Name;
            if (Version.TryParse(dirName.Substring(1), out var version))
            {
                return version;
            }
            return null;
        }

        private static string? FindCudart()
        {
            if (_cudaToolkitPath == null) return null;

            string binPath = Path.Combine(_cudaToolkitPath, "bin");
            if (Directory.Exists(binPath))
            {
                var dlls = Directory.GetFiles(binPath, "cudart64_*.dll")
                    .Select(path => new { Path = path, Version = GetVersionFromFileName(path) })
                    .OrderByDescending(x => x.Version)
                    .ToList();

                return dlls.FirstOrDefault()?.Path;
            }
            return null;
        }

        private static Version? GetVersionFromFileName(string filePath)
        {
            var fileName = Path.GetFileNameWithoutExtension(filePath);
            var parts = fileName.Split('_');
            if (parts.Length > 1 && int.TryParse(parts[1], out int majorVersion))
            {
                // simplified versioning, e.g., cudart64_12 -> 12.0
                return new Version(majorVersion, 0);
            }
            return null;
        }

        private static string? FindNvcomp()
        {
            if (_cudaToolkitPath == null) return null;

            string binPath = Path.Combine(_cudaToolkitPath, "bin");
            if (Directory.Exists(binPath))
            {
                string nvcompPath = Path.Combine(binPath, "nvcomp.dll");
                if (File.Exists(nvcompPath))
                {
                    return nvcompPath;
                }
            }
            return null;
        }

        public static string? CudartDllPath => _cudartDllPath;
        public static string? NvcompDllPath => _nvcompDllPath;

        public static bool AreLibrariesAvailable()
        {
            return !string.IsNullOrEmpty(_cudartDllPath) && !string.IsNullOrEmpty(_nvcompDllPath);
        }
    }
}
