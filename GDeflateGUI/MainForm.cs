using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace GDeflateGUI
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();

            // Wire up event handlers
            this.btnAddFiles.Click += new System.EventHandler(this.btnAddFiles_Click);
            this.btnAddFolder.Click += new System.EventHandler(this.btnAddFolder_Click);
            this.btnClear.Click += new System.EventHandler(this.btnClear_Click);
            this.btnCompress.Click += new System.EventHandler(this.btnCompress_Click);
            this.btnDecompress.Click += new System.EventHandler(this.btnDecompress_Click);
        }

        private async void btnDecompress_Click(object? sender, System.EventArgs e)
        {
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
            using (var dialog = new OpenFileDialog())
            {
                dialog.Multiselect = true;
                dialog.Title = "Select files";
                dialog.Filter = "All files (*.*)|*.*";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    foreach (string file in dialog.FileNames)
                    {
                        AddFileToList(file);
                    }
                    UpdateStatus($"Added {dialog.FileNames.Length} files. Total: {listViewFiles.Items.Count}");
                }
            }
        }

        private void btnAddFolder_Click(object? sender, System.EventArgs e)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Select a folder to add all its files";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    var files = Directory.GetFiles(dialog.SelectedPath, "*.*", SearchOption.AllDirectories);
                    foreach (string file in files)
                    {
                        AddFileToList(file);
                    }
                    UpdateStatus($"Added {files.Length} files from folder. Total: {listViewFiles.Items.Count}");
                }
            }
        }

        private void btnClear_Click(object? sender, System.EventArgs e)
        {
            listViewFiles.Items.Clear();
            UpdateStatus("File list cleared.");
        }

        private void AddFileToList(string filePath)
        {
            // Check if the file is already in the list to avoid duplicates
            if (!listViewFiles.Items.Cast<ListViewItem>().Any(item => item.Text == filePath))
            {
                var item = new ListViewItem(filePath);
                listViewFiles.Items.Add(item);
            }
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
