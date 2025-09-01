using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace GDeflateGUI
{
    public class ArchiveManager
    {
        public void CreateZipArchive(string archivePath, IEnumerable<string> files)
        {
            using (var fileStream = new FileStream(archivePath, FileMode.Create))
            using (var archive = new ZipArchive(fileStream, ZipArchiveMode.Create))
            {
                foreach (var file in files)
                {
                    archive.CreateEntryFromFile(file, Path.GetFileName(file));
                }
            }
        }

        public void ExtractZipArchive(string archivePath, string outputDirectory, Action<string, string> decompressFileAction)
        {
            using (var archive = ZipFile.OpenRead(archivePath))
            {
                bool hasGdefFiles = false;
                foreach (var entry in archive.Entries)
                {
                    string outputPath = Path.Combine(outputDirectory, entry.FullName);
                    Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

                    if (entry.Name.EndsWith(".gdef", StringComparison.OrdinalIgnoreCase))
                    {
                        hasGdefFiles = true;
                        // Create a temporary file for the gdef stream
                        string tempGdefFile = Path.GetTempFileName();
                        entry.ExtractToFile(tempGdefFile, true);

                        // Decompress the gdef file to its final destination
                        string finalPath = Path.ChangeExtension(outputPath, null);
                        decompressFileAction(tempGdefFile, finalPath);

                        File.Delete(tempGdefFile);
                    }
                    else
                    {
                        entry.ExtractToFile(outputPath, true);
                    }
                }

                if (!hasGdefFiles)
                {
                    throw new InvalidDataException("The selected .zip archive does not contain any .gdef files to decompress.");
                }
            }
        }
    }
}
