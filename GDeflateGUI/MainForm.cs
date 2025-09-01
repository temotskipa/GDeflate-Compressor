using System.IO;
using System.Linq;
using System.Windows.Forms;
using System.Runtime.InteropServices;

namespace GDeflateGUI
{
    public partial class MainForm : Form
    {
        private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static bool IsLinux => RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
        private static bool IsMacOS => RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

        public MainForm()
        {
            InitializeComponent();

            // Wire up event handlers
            this.btnAddFiles.Click += new System.EventHandler(this.btnAddFiles_Click);
            this.btnAddFolder.Click += new System.EventHandler(this.btnAddFolder_Click);
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            this.btnCompress.Click += new System.EventHandler(this.btnCompress_Click);
            this.btnDecompress.Click += new System.EventHandler(this.btnDecompress_Click);

            // Show OS info in status
            UpdateStatus($"Ready - Running on {RuntimeInformation.OSDescription}");
        }

        private async void btnDecompress_Click(object? sender, System.EventArgs e)
        {
            try
            {
                if (!IsWindows)
                {
                    MessageBox.Show("Decompression file selection is currently only supported on Windows.\n\nOn non-Windows platforms, please use the console version or run on Windows.", 
                        "Platform Limitation", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                using (var dialog = new OpenFileDialog())
                {
                    dialog.Multiselect = true;
                    dialog.Title = "Select files to decompress";
                    dialog.Filter = "GDeflate files (*.gdef)|*.gdef";
                    if (dialog.ShowDialog() != DialogResult.OK)
                    {
                        return;
                    }

                SetUIEnabled(false);
                UpdateStatus("Starting decompression...");

                var processor = new GDeflateProcessor();
                int successCount = 0;

                await System.Threading.Tasks.Task.Run(() =>
                {
                    for (int i = 0; i < dialog.FileNames.Length; i++)
                    {
                        string inputFile = dialog.FileNames[i];
                        string outputFile = Path.ChangeExtension(inputFile, null); // Removes .gdef
                        try
                        {
                            this.Invoke((MethodInvoker)delegate {
                                UpdateStatus($"Decompressing ({i + 1}/{dialog.FileNames.Length}): {Path.GetFileName(inputFile)}");
                            });

                            processor.DecompressFile(inputFile, outputFile);
                            successCount++;
                        }
                        catch (Exception ex)
                        {
                            var result = MessageBox.Show($"Failed to decompress {inputFile}.\nError: {ex.Message}\n\nContinue with next files?", "Decompression Error", MessageBoxButtons.YesNo, MessageBoxIcon.Error);
                            if (result == DialogResult.No)
                            {
                                break;
                            }
                        }
                    }
                });

                UpdateStatus($"Decompression finished. {successCount} out of {dialog.FileNames.Length} files decompressed successfully.");
                SetUIEnabled(true);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during decompression: {ex.Message}", "Decompression Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                UpdateStatus("Decompression failed due to error.");
                SetUIEnabled(true);
            }
        }

        private async void btnCompress_Click(object? sender, System.EventArgs e)
        {
            if (listViewFiles.Items.Count == 0)
            {
                MessageBox.Show("Please add files to compress.", "No Files", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            SetUIEnabled(false);
            UpdateStatus("Starting compression...");

            var processor = new GDeflateProcessor();
            var filesToCompress = listViewFiles.Items.Cast<ListViewItem>().Select(item => item.Text).ToList();
            int successCount = 0;

            await System.Threading.Tasks.Task.Run(() =>
            {
                for (int i = 0; i < filesToCompress.Count; i++)
                {
                    string inputFile = filesToCompress[i];
                    string outputFile = inputFile + ".gdef";
                    try
                    {
                        this.Invoke((MethodInvoker)delegate {
                            UpdateStatus($"Compressing ({i + 1}/{filesToCompress.Count}): {Path.GetFileName(inputFile)}");
                        });

                        processor.CompressFile(inputFile, outputFile);
                        successCount++;
                    }
                    catch (Exception ex)
                    {
                        var result = MessageBox.Show($"Failed to compress {inputFile}.\nError: {ex.Message}\n\nContinue with next files?", "Compression Error", MessageBoxButtons.YesNo, MessageBoxIcon.Error);
                        if (result == DialogResult.No)
                        {
                            break;
                        }
                    }
                }
            });

            UpdateStatus($"Compression finished. {successCount} out of {filesToCompress.Count} files compressed successfully.");
            SetUIEnabled(true);
        }

        private void btnAddFiles_Click(object? sender, System.EventArgs e)
        {
            if (!IsWindows)
            {
                AddFilesAlternativeMethod();
                return;
            }

            try
            {
                using (var dialog = new OpenFileDialog
                {
                    Multiselect = true,
                    Title = "Select files",
                    Filter = "All files (*.*)|*.*"
                })
                {
                    if (dialog.ShowDialog() == DialogResult.OK)
                    {
                        AddFilesToList(dialog.FileNames);
                    }
                }
            }
            catch (Exception ex)
            {
                ShowError("Error adding files", ex);
            }
        }

        private async void btnAddFolder_Click(object? sender, System.EventArgs e)
        {
            if (!IsWindows)
            {
                AddFolderAlternativeMethod();
                return;
            }

            try
            {
                using (var dialog = new FolderBrowserDialog
                {
                    Description = "Select a folder to add all its files"
                })
                {
                    if (dialog.ShowDialog() == DialogResult.OK)
                    {
                        await AddFolderFilesAsync(dialog.SelectedPath);
                    }
                }
            }
            catch (Exception ex)
            {
                ShowError("Error adding folder", ex);
            }
        }

        private async System.Threading.Tasks.Task AddFolderFilesAsync(string path)
        {
            SetUIEnabled(false);
            UpdateStatus("Searching for files...");

            try
            {
                var files = await System.Threading.Tasks.Task.Run(() =>
                    Directory.GetFiles(path, "*.*", SearchOption.AllDirectories)
                );
                AddFilesToList(files);
            }
            catch (UnauthorizedAccessException ex)
            {
                ShowError("Access denied. You may not have permission to access all subdirectories.", ex);
            }
            catch (IOException ex)
            {
                ShowError("An I/O error occurred while accessing the folder.", ex);
            }
            catch (Exception ex)
            {
                ShowError("An unexpected error occurred while adding folder files.", ex);
            }
            finally
            {
                SetUIEnabled(true);
            }
        }

        private void btnClear_Click(object? sender, System.EventArgs e)
        {
            listViewFiles.Items.Clear();
            UpdateStatus("File list cleared.");
        }

        private void AddFilesAlternativeMethod()
        {
            // For non-Windows platforms, show a message with instructions
            var result = MessageBox.Show(
                "File selection dialogs are not available on this platform.\n\n" +
                "Alternative methods:\n" +
                "1. Use the console version of this application\n" +
                "2. Add files programmatically\n" +
                "3. Run on Windows for full GUI support\n\n" +
                "Would you like to add some test files from the current directory?",
                "Alternative File Selection",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Information);

            if (result == DialogResult.Yes)
            {
                try
                {
                    // Add some test files from current directory
                    var currentDir = Directory.GetCurrentDirectory();
                    var files = Directory.GetFiles(currentDir, "*.*", SearchOption.TopDirectoryOnly)
                        .Take(5) // Limit to first 5 files
                        .ToArray();

                    AddFilesToList(files);
                }
                catch (Exception ex)
                {
                    ShowError("Error adding test files", ex);
                }
            }
        }

        private void AddFolderAlternativeMethod()
        {
            // For non-Windows platforms, show a message with instructions
            var result = MessageBox.Show(
                "Folder selection dialogs are not available on this platform.\n\n" +
                "Alternative methods:\n" +
                "1. Use the console version of this application\n" +
                "2. Add folders programmatically\n" +
                "3. Run on Windows for full GUI support\n\n" +
                "Would you like to add all files from the current directory?",
                "Alternative Folder Selection",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Information);

            if (result == DialogResult.Yes)
            {
                try
                {
                    var currentDir = Directory.GetCurrentDirectory();
                    var files = Directory.GetFiles(currentDir, "*.*", SearchOption.AllDirectories);
                    AddFilesToList(files);
                }
                catch (Exception ex)
                {
                    ShowError("Error adding files from folder", ex);
                }
            }
        }

        private void AddFilesToList(string[] filePaths)
        {
            var newItems = filePaths
                .Where(filePath => !listViewFiles.Items.Cast<ListViewItem>().Any(item => item.Text == filePath))
                .Select(filePath => new ListViewItem(filePath))
                .ToArray();

            if (newItems.Any())
            {
                listViewFiles.Items.AddRange(newItems);
                UpdateStatus($"Added {newItems.Length} files. Total: {listViewFiles.Items.Count}");
            }
        }

        private void ShowError(string title, Exception ex)
        {
            MessageBox.Show($"{ex.Message}\n\nTip: On non-Windows platforms, you can use the console version for more detailed error information.",
                title, MessageBoxButtons.OK, MessageBoxIcon.Warning);
            UpdateStatus($"Error: {title}. See message for details.");
        }

        private void UpdateStatus(string text)
        {
            statusLabel.Text = text;
        }

        private void SetUIEnabled(bool enabled)
        {
            this.btnAddFiles.Enabled = enabled;
            this.btnAddFolder.Enabled = enabled;
            this.btnClear.Enabled = enabled;
            this.btnCompress.Enabled = enabled;
            this.btnDecompress.Enabled = enabled;
        }
    }
}
