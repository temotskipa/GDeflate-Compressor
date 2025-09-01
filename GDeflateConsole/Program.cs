using System;
using System.IO;
using System.Runtime.InteropServices;

namespace GDeflateConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("GDeflate Console Application");
            Console.WriteLine($"Running on: {RuntimeInformation.OSDescription}");
            Console.WriteLine($"Architecture: {RuntimeInformation.OSArchitecture}");

            var processor = new GDeflateProcessor();
            Console.WriteLine($"GPU Support: {(processor.IsGpuAvailable() ? "Available" : "Not Available")}");
            Console.WriteLine($"Mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");
            Console.WriteLine();

            if (args.Length == 0)
            {
                ShowUsage();
                return;
            }

            try
            {
                var command = args[0].ToLower();
                var arguments = args.Skip(1).ToList();

                switch (command)
                {
                    case "compress":
                        ParseCompressCommand(arguments, processor);
                        break;
                    case "decompress":
                        ParseDecompressCommand(arguments, processor);
                        break;
                    case "list":
                        if (arguments.Count < 1) throw new ArgumentException("list command requires a directory path.");
                        ListFiles(arguments[0]);
                        break;
                    case "test":
                        RunTests(processor);
                        break;
                    default:
                        Console.WriteLine($"Error: Unknown command '{command}'");
                        ShowUsage();
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                }
                Environment.Exit(1);
            }
        }

        static void ShowUsage()
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("  GDeflateConsole compress <file1> [file2...] [--format <zip|gdef>] - Compress file(s)");
            Console.WriteLine("  GDeflateConsole decompress <file> [--output-dir <path>]          - Decompress a .gdef or .zip file");
            Console.WriteLine("  GDeflateConsole list <directory>                                 - List files in a directory");
            Console.WriteLine("  GDeflateConsole test                                             - Run compression/decompression tests");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --format <zip|gdef>   - Output format for compression. Default is gdef for single files, zip for multiple.");
            Console.WriteLine("  --output-dir <path>   - Directory to extract files to. Default is a new folder named after the archive.");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  GDeflateConsole compress document.pdf");
            Console.WriteLine("  GDeflateConsole compress file1.txt file2.txt --format zip");
            Console.WriteLine("  GDeflateConsole decompress document.pdf.gdef");
            Console.WriteLine("  GDeflateConsole decompress my_archive.zip --output-dir ./extracted_files");
            Console.WriteLine();
            Console.WriteLine("Note: GPU acceleration is used when CUDA and nvCOMP are available.");
            Console.WriteLine("      Otherwise, the application runs in simulation mode for testing.");
        }

        static void ParseCompressCommand(List<string> args, GDeflateProcessor processor)
        {
            var inputFiles = args.Where(a => !a.StartsWith("--")).ToList();
            if (inputFiles.Count == 0) throw new ArgumentException("compress command requires at least one file path.");

            string format = "gdef";
            int formatIndex = args.IndexOf("--format");
            if (formatIndex != -1 && args.Count > formatIndex + 1)
            {
                format = args[formatIndex + 1].ToLower();
            }
            else if (inputFiles.Count > 1)
            {
                format = "zip";
            }

            string outputFormatExtension = "." + format;
            string outputFileName = inputFiles.Count > 1 ? "archive.zip" : Path.GetFileName(inputFiles[0]) + outputFormatExtension;
            string outputPath = Path.Combine(Directory.GetCurrentDirectory(), outputFileName);

            Compress(inputFiles.ToArray(), outputPath, outputFormatExtension, processor);
        }

        static void ParseDecompressCommand(List<string> args, GDeflateProcessor processor)
        {
            var inputFile = args.FirstOrDefault(a => !a.StartsWith("--"));
            if (string.IsNullOrEmpty(inputFile)) throw new ArgumentException("decompress command requires a file path.");

            string outputDir = "";
            int outputDirIndex = args.IndexOf("--output-dir");
            if (outputDirIndex != -1 && args.Count > outputDirIndex + 1)
            {
                outputDir = args[outputDirIndex + 1];
            }
            else
            {
                outputDir = Path.Combine(Directory.GetCurrentDirectory(), Path.GetFileNameWithoutExtension(inputFile));
            }
            Directory.CreateDirectory(outputDir);

            Decompress(inputFile, outputDir, processor);
        }

        static void Compress(string[] filePaths, string outputPath, string format, GDeflateProcessor processor)
        {
            foreach (var filePath in filePaths)
            {
                if (!File.Exists(filePath)) throw new FileNotFoundException($"File not found: {filePath}");
            }

            Console.WriteLine($"Compressing {filePaths.Length} file(s)...");
            Console.WriteLine($"Output: {outputPath}");
            Console.WriteLine($"Mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");

            var startTime = DateTime.Now;
            try
            {
                processor.CompressFilesToArchive(filePaths, outputPath, format);
                var endTime = DateTime.Now;
                Console.WriteLine($"Compression completed successfully in { (endTime - startTime).TotalMilliseconds:F2} ms!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Compression failed: {ex.Message}");
                if (File.Exists(outputPath))
                {
                    try { File.Delete(outputPath); } catch { }
                }
                throw;
            }
        }

        static void Decompress(string filePath, string outputDir, GDeflateProcessor processor)
        {
            if (!File.Exists(filePath)) throw new FileNotFoundException($"File not found: {filePath}");

            Console.WriteLine($"Decompressing: {filePath}");
            Console.WriteLine($"Output Directory: {outputDir}");
            Console.WriteLine($"Mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");
            
            var startTime = DateTime.Now;
            try
            {
                processor.DecompressArchive(filePath, outputDir);
                var endTime = DateTime.Now;
                Console.WriteLine($"Decompression completed successfully in {(endTime - startTime).TotalMilliseconds:F2} ms!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Decompression failed: {ex.Message}");
                throw;
            }
        }

        static void ListFiles(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
            {
                throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");
            }

            Console.WriteLine($"Files in directory: {directoryPath}");
            Console.WriteLine();

            var files = Directory.GetFiles(directoryPath);
            var directories = Directory.GetDirectories(directoryPath);

            // List directories first
            foreach (var dir in directories)
            {
                var dirInfo = new DirectoryInfo(dir);
                Console.WriteLine($"[DIR]  {dirInfo.Name}");
            }

            // List files
            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                string sizeStr = FormatFileSize(fileInfo.Length);
                string extension = fileInfo.Extension.ToUpper();
                
                if (extension == ".GDEF")
                {
                    Console.WriteLine($"[GDEF] {fileInfo.Name} ({sizeStr})");
                }
                else
                {
                    Console.WriteLine($"[FILE] {fileInfo.Name} ({sizeStr})");
                }
            }

            Console.WriteLine();
            Console.WriteLine($"Total: {directories.Length} directories, {files.Length} files");
        }

        static void RunTests(GDeflateProcessor processor)
        {
            Console.WriteLine("Running compression/decompression tests...");
            Console.WriteLine($"Test mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");
            Console.WriteLine();

            // Create test data
            string testDir = Path.Combine(Path.GetTempPath(), "gdeflate_test");
            Directory.CreateDirectory(testDir);

            try
            {
                // Test 1: Small text file
                Console.WriteLine("Test 1: Small text file");
                string testFile1 = Path.Combine(testDir, "test1.txt");
                string testData1 = "Hello, World! This is a test file for GDeflate compression. " +
                                  "It contains some repeated text to test compression efficiency. " +
                                  "Hello, World! This is a test file for GDeflate compression.";
                File.WriteAllText(testFile1, testData1);
                
                TestCompressionDecompression(testFile1, processor);
                Console.WriteLine();

                // Test 2: Binary data
                Console.WriteLine("Test 2: Binary data");
                string testFile2 = Path.Combine(testDir, "test2.bin");
                var binaryData = new byte[1024];
                for (int i = 0; i < binaryData.Length; i++)
                {
                    binaryData[i] = (byte)(i % 256);
                }
                File.WriteAllBytes(testFile2, binaryData);
                
                TestCompressionDecompression(testFile2, processor);
                Console.WriteLine();

                Console.WriteLine("All tests completed successfully!");
            }
            finally
            {
                // Clean up test directory
                try
                {
                    Directory.Delete(testDir, true);
                }
                catch
                {
                    Console.WriteLine($"Warning: Could not clean up test directory: {testDir}");
                }
            }
        }

        static void TestCompressionDecompression(string testFile, GDeflateProcessor processor)
        {
            string compressedFile = testFile + ".gdef";
            string decompressedFile = testFile + ".decompressed";

            try
            {
                var originalInfo = new FileInfo(testFile);
                Console.WriteLine($"  Original size: {FormatFileSize(originalInfo.Length)}");

                // Compress
                var compressStart = DateTime.Now;
                processor.CompressFile(testFile, compressedFile);
                var compressTime = DateTime.Now - compressStart;

                var compressedInfo = new FileInfo(compressedFile);
                Console.WriteLine($"  Compressed size: {FormatFileSize(compressedInfo.Length)}");
                Console.WriteLine($"  Compression ratio: {(double)compressedInfo.Length / originalInfo.Length:P2}");
                Console.WriteLine($"  Compression time: {compressTime.TotalMilliseconds:F2} ms");

                // Decompress
                var decompressStart = DateTime.Now;
                processor.DecompressFile(compressedFile, decompressedFile);
                var decompressTime = DateTime.Now - decompressStart;

                var decompressedInfo = new FileInfo(decompressedFile);
                Console.WriteLine($"  Decompressed size: {FormatFileSize(decompressedInfo.Length)}");
                Console.WriteLine($"  Decompression time: {decompressTime.TotalMilliseconds:F2} ms");

                // Verify integrity
                Console.WriteLine("  Verifying file integrity...");
                bool filesMatch = AreFilesEqual(testFile, decompressedFile);
                Console.WriteLine($"  Integrity check: {(filesMatch ? "PASSED" : "FAILED")}");

                if (!filesMatch)
                {
                    throw new Exception("File integrity check failed: decompressed file does not match original.");
                }

                Console.WriteLine("  Test PASSED");
            }
            finally
            {
                // Clean up test files
                try
                {
                    if (File.Exists(compressedFile)) File.Delete(compressedFile);
                    if (File.Exists(decompressedFile)) File.Delete(decompressedFile);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        static bool AreFilesEqual(string path1, string path2)
        {
            const int bufferSize = 4096;

            using (var fs1 = new FileStream(path1, FileMode.Open, FileAccess.Read))
            using (var fs2 = new FileStream(path2, FileMode.Open, FileAccess.Read))
            {
                if (fs1.Length != fs2.Length)
                {
                    return false;
                }

                var buffer1 = new byte[bufferSize];
                var buffer2 = new byte[bufferSize];

                while (true)
                {
                    int bytesRead1 = fs1.Read(buffer1, 0, bufferSize);
                    int bytesRead2 = fs2.Read(buffer2, 0, bufferSize);

                    if (bytesRead1 != bytesRead2)
                    {
                        return false;
                    }

                    if (bytesRead1 == 0)
                    {
                        return true;
                    }

                    if (!buffer1.AsSpan(0, bytesRead1).SequenceEqual(buffer2.AsSpan(0, bytesRead2)))
                    {
                        return false;
                    }
                }
            }
        }

        static string FormatFileSize(long bytes)
        {
            string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
            int counter = 0;
            decimal number = bytes;
            
            while (Math.Round(number / 1024) >= 1)
            {
                number /= 1024;
                counter++;
            }
            
            return $"{number:N1} {suffixes[counter]}";
        }
    }
}