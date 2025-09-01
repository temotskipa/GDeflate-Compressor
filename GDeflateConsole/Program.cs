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
            
            // Initialize GPU processor
            var processor = new GDeflateProcessor();
            
            Console.WriteLine($"GPU Support: {(processor.IsGpuAvailable() ? "Available" : "Not Available")}");
            Console.WriteLine($"Mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");
            Console.WriteLine();
            
            if (args.Length == 0)
            {
                ShowUsage();
                return;
            }

            string command = args[0].ToLower();
            
            try
            {
                switch (command)
                {
                    case "compress":
                        if (args.Length < 2)
                        {
                            Console.WriteLine("Error: compress command requires a file path");
                            ShowUsage();
                            return;
                        }
                        CompressFile(args[1], processor);
                        break;
                        
                    case "decompress":
                        if (args.Length < 2)
                        {
                            Console.WriteLine("Error: decompress command requires a file path");
                            ShowUsage();
                            return;
                        }
                        DecompressFile(args[1], processor);
                        break;
                        
                    case "list":
                        if (args.Length < 2)
                        {
                            Console.WriteLine("Error: list command requires a directory path");
                            ShowUsage();
                            return;
                        }
                        ListFiles(args[1]);
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
            Console.WriteLine("  GDeflateConsole compress <file>     - Compress a file");
            Console.WriteLine("  GDeflateConsole decompress <file>   - Decompress a .gdef file");
            Console.WriteLine("  GDeflateConsole list <directory>    - List files in directory");
            Console.WriteLine("  GDeflateConsole test                - Run compression/decompression tests");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  GDeflateConsole compress document.pdf");
            Console.WriteLine("  GDeflateConsole decompress document.pdf.gdef");
            Console.WriteLine("  GDeflateConsole list /path/to/files");
            Console.WriteLine("  GDeflateConsole test");
            Console.WriteLine();
            Console.WriteLine("Note: GPU acceleration is used when CUDA and nvCOMP are available.");
            Console.WriteLine("      Otherwise, the application runs in simulation mode for testing.");
        }

        static void CompressFile(string filePath, GDeflateProcessor processor)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            string outputPath = filePath + ".gdef";
            
            Console.WriteLine($"Compressing: {filePath}");
            Console.WriteLine($"Output: {outputPath}");
            Console.WriteLine($"Mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");
            
            // Get input file size
            var fileInfo = new FileInfo(filePath);
            Console.WriteLine($"Input size: {FormatFileSize(fileInfo.Length)}");
            
            var startTime = DateTime.Now;
            
            try
            {
                processor.CompressFile(filePath, outputPath);
                
                var endTime = DateTime.Now;
                var duration = endTime - startTime;
                
                var outputInfo = new FileInfo(outputPath);
                Console.WriteLine($"Compressed size: {FormatFileSize(outputInfo.Length)}");
                Console.WriteLine($"Compression ratio: {(double)outputInfo.Length / fileInfo.Length:P2}");
                Console.WriteLine($"Processing time: {duration.TotalMilliseconds:F2} ms");
                Console.WriteLine("Compression completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Compression failed: {ex.Message}");
                
                // Clean up partial output file if it exists
                if (File.Exists(outputPath))
                {
                    try
                    {
                        File.Delete(outputPath);
                    }
                    catch
                    {
                        // Ignore cleanup errors
                    }
                }
                throw;
            }
        }

        static void DecompressFile(string filePath, GDeflateProcessor processor)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            if (!filePath.EndsWith(".gdef", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException("File must have .gdef extension");
            }

            string outputPath = filePath.Substring(0, filePath.Length - 5); // Remove .gdef extension
            
            Console.WriteLine($"Decompressing: {filePath}");
            Console.WriteLine($"Output: {outputPath}");
            Console.WriteLine($"Mode: {(processor.IsSimulationMode ? "Simulation" : "GPU Accelerated")}");
            
            // Get compressed file size
            var fileInfo = new FileInfo(filePath);
            Console.WriteLine($"Compressed size: {FormatFileSize(fileInfo.Length)}");
            
            var startTime = DateTime.Now;
            
            try
            {
                processor.DecompressFile(filePath, outputPath);
                
                var endTime = DateTime.Now;
                var duration = endTime - startTime;
                
                var outputInfo = new FileInfo(outputPath);
                Console.WriteLine($"Decompressed size: {FormatFileSize(outputInfo.Length)}");
                Console.WriteLine($"Processing time: {duration.TotalMilliseconds:F2} ms");
                Console.WriteLine("Decompression completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Decompression failed: {ex.Message}");
                
                // Clean up partial output file if it exists
                if (File.Exists(outputPath))
                {
                    try
                    {
                        File.Delete(outputPath);
                    }
                    catch
                    {
                        // Ignore cleanup errors
                    }
                }
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