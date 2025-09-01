using System;
using System.IO;
using System.Linq;

namespace GDeflateConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("GDeflate Console - Cross-Platform File Compression Tool");
            Console.WriteLine("======================================================");
            Console.WriteLine($"Running on: {System.Runtime.InteropServices.RuntimeInformation.OSDescription}");
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
                    case "c":
                        HandleCompress(args.Skip(1).ToArray());
                        break;
                    case "decompress":
                    case "d":
                        HandleDecompress(args.Skip(1).ToArray());
                        break;
                    case "list":
                    case "l":
                        HandleList(args.Skip(1).ToArray());
                        break;
                    default:
                        Console.WriteLine($"Unknown command: {command}");
                        ShowUsage();
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Environment.Exit(1);
            }
        }

        static void ShowUsage()
        {
            Console.WriteLine("Usage:");
            Console.WriteLine("  GDeflateConsole compress <input-file> [output-file]");
            Console.WriteLine("  GDeflateConsole decompress <input-file> [output-file]");
            Console.WriteLine("  GDeflateConsole list <directory>");
            Console.WriteLine();
            Console.WriteLine("Commands:");
            Console.WriteLine("  compress, c    - Compress a file");
            Console.WriteLine("  decompress, d  - Decompress a .gdef file");
            Console.WriteLine("  list, l        - List files in directory for compression");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  GDeflateConsole compress myfile.txt");
            Console.WriteLine("  GDeflateConsole decompress myfile.txt.gdef");
            Console.WriteLine("  GDeflateConsole list /path/to/directory");
        }

        static void HandleCompress(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Error: Input file required for compression");
                return;
            }

            string inputFile = args[0];
            string outputFile = args.Length > 1 ? args[1] : inputFile + ".gdef";

            if (!File.Exists(inputFile))
            {
                Console.WriteLine($"Error: Input file not found: {inputFile}");
                return;
            }

            Console.WriteLine($"Compressing: {inputFile} -> {outputFile}");

            try
            {
                // Note: This would normally use GDeflateProcessor, but since we're on Linux
                // and don't have CUDA/nvCOMP, we'll simulate the operation
                Console.WriteLine("Warning: CUDA/nvCOMP not available on this platform.");
                Console.WriteLine("This is a simulation - actual compression requires Windows with NVIDIA GPU.");
                
                // Simulate compression by copying the file with .gdef extension
                File.Copy(inputFile, outputFile, true);
                
                var inputSize = new FileInfo(inputFile).Length;
                var outputSize = new FileInfo(outputFile).Length;
                
                Console.WriteLine($"Compression completed (simulated)");
                Console.WriteLine($"Input size: {inputSize:N0} bytes");
                Console.WriteLine($"Output size: {outputSize:N0} bytes");
                Console.WriteLine($"Compression ratio: {(double)outputSize / inputSize:P2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Compression failed: {ex.Message}");
            }
        }

        static void HandleDecompress(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Error: Input file required for decompression");
                return;
            }

            string inputFile = args[0];
            string outputFile = args.Length > 1 ? args[1] : Path.ChangeExtension(inputFile, null);

            if (!File.Exists(inputFile))
            {
                Console.WriteLine($"Error: Input file not found: {inputFile}");
                return;
            }

            Console.WriteLine($"Decompressing: {inputFile} -> {outputFile}");

            try
            {
                // Note: This would normally use GDeflateProcessor, but since we're on Linux
                // and don't have CUDA/nvCOMP, we'll simulate the operation
                Console.WriteLine("Warning: CUDA/nvCOMP not available on this platform.");
                Console.WriteLine("This is a simulation - actual decompression requires Windows with NVIDIA GPU.");
                
                // Simulate decompression by copying the file without .gdef extension
                File.Copy(inputFile, outputFile, true);
                
                var inputSize = new FileInfo(inputFile).Length;
                var outputSize = new FileInfo(outputFile).Length;
                
                Console.WriteLine($"Decompression completed (simulated)");
                Console.WriteLine($"Input size: {inputSize:N0} bytes");
                Console.WriteLine($"Output size: {outputSize:N0} bytes");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Decompression failed: {ex.Message}");
            }
        }

        static void HandleList(string[] args)
        {
            string directory = args.Length > 0 ? args[0] : Directory.GetCurrentDirectory();

            if (!Directory.Exists(directory))
            {
                Console.WriteLine($"Error: Directory not found: {directory}");
                return;
            }

            Console.WriteLine($"Files in directory: {directory}");
            Console.WriteLine(new string('-', 50));

            try
            {
                var files = Directory.GetFiles(directory, "*.*", SearchOption.TopDirectoryOnly)
                    .OrderBy(f => f)
                    .ToArray();

                if (files.Length == 0)
                {
                    Console.WriteLine("No files found.");
                    return;
                }

                for (int i = 0; i < files.Length; i++)
                {
                    var fileInfo = new FileInfo(files[i]);
                    Console.WriteLine($"{i + 1,3}. {Path.GetFileName(files[i])} ({fileInfo.Length:N0} bytes)");
                }

                Console.WriteLine();
                Console.WriteLine($"Total: {files.Length} files");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error listing files: {ex.Message}");
            }
        }
    }
}