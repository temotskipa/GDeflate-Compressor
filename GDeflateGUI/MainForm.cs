using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using System.Runtime.InteropServices;

namespace GDeflateGUI
{
    public partial class MainForm : Form
    {
        private static bool IsWindows => RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private ComboBox _formatComboBox;
        private Label _gpuStatusLabel;
        private GDeflateProcessor _processor;

        public MainForm()
        {
            InitializeComponent();
            InitializeCustomComponents();

            _processor = new GDeflateProcessor();
            UpdateGpuStatus();

            this.btnAddFiles.Click += new System.EventHandler(this.btnAddFiles_Click);
            this.btnAddFolder.Click += new System.EventHandler(this.btnAddFolder_Click);
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            this.btnCompress.Click += new System.EventHandler(this.btnCompress_Click);
            this.btnDecompress.Click += new System.EventHandler(this.btnDecompress_Click);

            UpdateStatus($"Ready - Running on {RuntimeInformation.OSDescription}");
        }

        private void InitializeCustomComponents()
        {
            // Format ComboBox
            _formatComboBox = new ComboBox();
            _formatComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            _formatComboBox.FormattingEnabled = true;
            _formatComboBox.Items.AddRange(new object[] { ".gdef (single file)", ".zip (archive)" });
            _formatComboBox.Location = new System.Drawing.Point(588, 280);
            _formatComboBox.Name = "_formatComboBox";
            _formatComboBox.Size = new System.Drawing.Size(184, 23);
            _formatComboBox.TabIndex = 7;
            _formatComboBox.SelectedIndex = 0;
            Controls.Add(_formatComboBox);

            var formatLabel = new Label();
            formatLabel.Text = "Output Format:";
            formatLabel.Location = new Point(588, 260);
            formatLabel.Size = new Size(100, 20);
            Controls.Add(formatLabel);

            // GPU Status Label
            _gpuStatusLabel = new Label();
            _gpuStatusLabel.Location = new Point(588, 150);
            _gpuStatusLabel.Name = "_gpuStatusLabel";
            _gpuStatusLabel.Size = new System.Drawing.Size(184, 40);
            _gpuStatusLabel.Font = new Font("Segoe UI", 9F, FontStyle.Bold);
            Controls.Add(_gpuStatusLabel);
        }

        private void UpdateGpuStatus()
        {
            if (_processor.IsGpuAvailable())
            {
                _gpuStatusLabel.Text = "GPU: Active";
                _gpuStatusLabel.ForeColor = Color.Green;
            }
            else
            {
                _gpuStatusLabel.Text = "GPU: Not Available\n(Running in Simulation Mode)";
                _gpuStatusLabel.ForeColor = Color.Red;
            }
        }

        private async void btnDecompress_Click(object? sender, System.EventArgs e)
        {
            try
            {
                using (var dialog = new OpenFileDialog())
                {
                    dialog.Multiselect = true;
                    dialog.Title = "Select files to decompress";
                    dialog.Filter = "Archives (*.gdef, *.zip)|*.gdef;*.zip|All files (*.*)|*.*";
                    if (dialog.ShowDialog() != DialogResult.OK) return;

                    string outputDir = "";
                    using (var folderDialog = new FolderBrowserDialog())
                    {
                        folderDialog.Description = "Select a folder to extract the files to.";
                        if (folderDialog.ShowDialog() != DialogResult.OK) return;
                        outputDir = folderDialog.SelectedPath;
                    }

                    SetUIEnabled(false);
                    UpdateStatus("Starting decompression...");
                    int successCount = 0;

                    await System.Threading.Tasks.Task.Run(() =>
                    {
                        for (int i = 0; i < dialog.FileNames.Length; i++)
                        {
                            string inputFile = dialog.FileNames[i];
                            try
                            {
                                this.Invoke((MethodInvoker)delegate {
                                    UpdateStatus($"Decompressing ({i + 1}/{dialog.FileNames.Length}): {Path.GetFileName(inputFile)}");
                                });
                                _processor.DecompressArchive(inputFile, outputDir);
                                successCount++;
                            }
                            catch (Exception ex)
                            {
                                var result = MessageBox.Show($"Failed to decompress {inputFile}.\nError: {ex.Message}\n\nContinue with next files?", "Decompression Error", MessageBoxButtons.YesNo, MessageBoxIcon.Error);
                                if (result == DialogResult.No) break;
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

            string format = _formatComboBox.SelectedIndex == 0 ? ".gdef" : ".zip";
            if (format == ".gdef" && listViewFiles.Items.Count > 1)
            {
                MessageBox.Show(".gdef format only supports compressing a single file.", "Format Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            using (var dialog = new SaveFileDialog())
            {
                dialog.Title = "Save archive as...";
                dialog.Filter = format == ".gdef" ? "GDeflate File (*.gdef)|*.gdef" : "Zip Archive (*.zip)|*.zip";
                dialog.FileName = format == ".gdef" ? Path.GetFileNameWithoutExtension(listViewFiles.Items[0].Text) + ".gdef" : "archive.zip";

                if (dialog.ShowDialog() != DialogResult.OK) return;

                SetUIEnabled(false);
                UpdateStatus("Starting compression...");

                var filesToCompress = listViewFiles.Items.Cast<ListViewItem>().Select(item => item.Text).ToArray();

                await System.Threading.Tasks.Task.Run(() =>
                {
                    try
                    {
                        _processor.CompressFilesToArchive(filesToCompress, dialog.FileName, format);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Failed to compress files.\nError: {ex.Message}", "Compression Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                });

                UpdateStatus($"Compression finished. Output: {dialog.FileName}");
                SetUIEnabled(true);
            }
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
                UpdateStatus($"Found {listViewFiles.Items.Count} files.");
            }
        }

        private void btnClear_Click(object? sender, System.EventArgs e)
        {
            listViewFiles.Items.Clear();
            UpdateStatus("File list cleared.");
        }

        private void AddFilesAlternativeMethod()
        {
            var result = MessageBox.Show(
                "File selection dialogs are not available on this platform.\n\n" +
                "Please use the console version for full functionality.",
                "Platform Limitation",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information);
        }

        private void AddFolderAlternativeMethod()
        {
            AddFilesAlternativeMethod();
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
            MessageBox.Show($"{ex.Message}", title, MessageBoxButtons.OK, MessageBoxIcon.Warning);
            UpdateStatus($"Error: {title}.");
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
            this._formatComboBox.Enabled = enabled;
        }
    }
}
