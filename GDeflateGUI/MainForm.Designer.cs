namespace GDeflateGUI
{
    partial class MainForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.listViewFiles = new System.Windows.Forms.ListView();
            this.colFilePath = new System.Windows.Forms.ColumnHeader();
            this.btnAddFiles = new System.Windows.Forms.Button();
            this.btnAddFolder = new System.Windows.Forms.Button();
            this.btnCompress = new System.Windows.Forms.Button();
            this.btnDecompress = new System.Windows.Forms.Button();
            this.btnClear = new System.Windows.Forms.Button();
            this.statusStrip = new System.Windows.Forms.StatusStrip();
            this.statusLabel = new System.Windows.Forms.ToolStripStatusLabel();
            this.statusStrip.SuspendLayout();
            this.SuspendLayout();
            //
            // listViewFiles
            //
            this.listViewFiles.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
            | System.Windows.Forms.AnchorStyles.Left)
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listViewFiles.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.colFilePath});
            this.listViewFiles.FullRowSelect = true;
            this.listViewFiles.Location = new System.Drawing.Point(12, 12);
            this.listViewFiles.Name = "listViewFiles";
            this.listViewFiles.Size = new System.Drawing.Size(558, 395);
            this.listViewFiles.TabIndex = 0;
            this.listViewFiles.UseCompatibleStateImageBehavior = false;
            this.listViewFiles.View = System.Windows.Forms.View.Details;
            //
            // colFilePath
            //
            this.colFilePath.Text = "File Path";
            this.colFilePath.Width = 550;
            //
            // btnAddFiles
            //
            this.btnAddFiles.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnAddFiles.Location = new System.Drawing.Point(588, 12);
            this.btnAddFiles.Name = "btnAddFiles";
            this.btnAddFiles.Size = new System.Drawing.Size(184, 35);
            this.btnAddFiles.TabIndex = 1;
            this.btnAddFiles.Text = "Add Files...";
            this.btnAddFiles.UseVisualStyleBackColor = true;
            //
            // btnAddFolder
            //
            this.btnAddFolder.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnAddFolder.Location = new System.Drawing.Point(588, 53);
            this.btnAddFolder.Name = "btnAddFolder";
            this.btnAddFolder.Size = new System.Drawing.Size(184, 35);
            this.btnAddFolder.TabIndex = 2;
            this.btnAddFolder.Text = "Add Folder...";
            this.btnAddFolder.UseVisualStyleBackColor = true;
            //
            // btnCompress
            //
            this.btnCompress.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCompress.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.btnCompress.Location = new System.Drawing.Point(588, 321);
            this.btnCompress.Name = "btnCompress";
            this.btnCompress.Size = new System.Drawing.Size(184, 40);
            this.btnCompress.TabIndex = 4;
            this.btnCompress.Text = "Compress";
            this.btnCompress.UseVisualStyleBackColor = true;
            //
            // btnDecompress
            //
            this.btnDecompress.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.btnDecompress.Location = new System.Drawing.Point(588, 367);
            this.btnDecompress.Name = "btnDecompress";
            this.btnDecompress.Size = new System.Drawing.Size(184, 40);
            this.btnDecompress.TabIndex = 5;
            this.btnDecompress.Text = "Decompress";
            this.btnDecompress.UseVisualStyleBackColor = true;
            //
            // btnClear
            //
            this.btnClear.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btnClear.Location = new System.Drawing.Point(588, 94);
            this.btnClear.Name = "btnClear";
            this.btnClear.Size = new System.Drawing.Size(184, 35);
            this.btnClear.TabIndex = 3;
            this.btnClear.Text = "Clear List";
            this.btnClear.UseVisualStyleBackColor = true;
            //
            // statusStrip
            //
            this.statusStrip.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.statusLabel});
            this.statusStrip.Location = new System.Drawing.Point(0, 428);
            this.statusStrip.Name = "statusStrip";
            this.statusStrip.Size = new System.Drawing.Size(784, 22);
            this.statusStrip.TabIndex = 6;
            this.statusStrip.Text = "statusStrip1";
            //
            // statusLabel
            //
            this.statusLabel.Name = "statusLabel";
            this.statusLabel.Size = new System.Drawing.Size(50, 17);
            this.statusLabel.Text = "Ready";
            //
            // MainForm
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(784, 450);
            this.Controls.Add(this.statusStrip);
            this.Controls.Add(this.btnClear);
            this.Controls.Add(this.btnDecompress);
            this.Controls.Add(this.btnCompress);
            this.Controls.Add(this.btnAddFolder);
            this.Controls.Add(this.btnAddFiles);
            this.Controls.Add(this.listViewFiles);
            this.MinimumSize = new System.Drawing.Size(400, 300);
            this.Name = "MainForm";
            this.Text = "GDeflate GUI";
            this.statusStrip.ResumeLayout(false);
            this.statusStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        #endregion

        private System.Windows.Forms.ListView listViewFiles;
        private System.Windows.Forms.Button btnAddFiles;
        private System.Windows.Forms.Button btnAddFolder;
        private System.Windows.Forms.Button btnCompress;
        private System.Windows.Forms.Button btnDecompress;
        private System.Windows.Forms.Button btnClear;
        private System.Windows.Forms.StatusStrip statusStrip;
        private System.Windows.Forms.ToolStripStatusLabel statusLabel;
        private System.Windows.Forms.ColumnHeader colFilePath;
    }
}
