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
                string[] programFilesPaths = {
                    Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                    Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86)
                };

                foreach (var programFilesPath in programFilesPaths.Distinct())
                {
                    if (string.IsNullOrEmpty(programFilesPath)) continue;

                    string nvidiaGpuComputingToolkit = Path.Combine(programFilesPath, "NVIDIA GPU Computing Toolkit", "CUDA");
                    if (Directory.Exists(nvidiaGpuComputingToolkit))
                    {
                        var versions = Directory.GetDirectories(nvidiaGpuComputingToolkit, "v*.*")
                            .Select(path => new { Path = path, Version = GetVersionFromPath(path) })
                            .OrderByDescending(x => x.Version)
                            .ToList();

                        var latestVersion = versions.FirstOrDefault();
                        if (latestVersion != null)
                        {
                            return latestVersion.Path;
                        }
                    }
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
            // Search in common paths first
            string[] searchPaths = { Directory.GetCurrentDirectory(), AppContext.BaseDirectory };
            foreach (var path in searchPaths)
            {
                var dlls = Directory.GetFiles(path, "cudart64_*.dll", SearchOption.AllDirectories)
                    .Select(p => new { Path = p, Version = GetVersionFromFileName(p) })
                    .OrderByDescending(x => x.Version)
                    .ToList();
                if (dlls.Any()) return dlls.First().Path;
            }

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
            // Search in common paths first
            string[] searchPaths = { Directory.GetCurrentDirectory(), AppContext.BaseDirectory };
            foreach (var path in searchPaths)
            {
                var nvcompPaths = Directory.GetFiles(path, "nvcomp.dll", SearchOption.AllDirectories);
                if (nvcompPaths.Any()) return nvcompPaths.First();
            }

            // Search in standard nvCOMP installation path on Windows
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                string[] programFilesPaths = {
                    Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                    Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86)
                };

                foreach (var programFilesPath in programFilesPaths.Distinct())
                {
                    if (string.IsNullOrEmpty(programFilesPath)) continue;

                    string nvcompInstallPath = Path.Combine(programFilesPath, "NVIDIA Corporation", "nvCOMP");
                    if (Directory.Exists(nvcompInstallPath))
                    {
                        var nvcompDllPaths = Directory.GetFiles(nvcompInstallPath, "nvcomp.dll", SearchOption.AllDirectories);
                        if (nvcompDllPaths.Any())
                        {
                            return nvcompDllPaths.First();
                        }
                    }
                }
            }

            if (_cudaToolkitPath == null) return null;

            string binPath = Path.Combine(_cudaToolkitPath, "bin");
            if (Directory.Exists(binPath))
            {
                // Search recursively for nvcomp.dll
                var nvcompPaths = Directory.GetFiles(binPath, "nvcomp.dll", SearchOption.AllDirectories);
                if (nvcompPaths.Length > 0)
                {
                    // Return the first match
                    return nvcompPaths[0];
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
