# GDeflate Compressor Repository

## Purpose
GDeflate Compressor is a Windows Forms desktop application that provides GPU-accelerated deflate compression and decompression using NVIDIA's nvCOMP library. The application leverages CUDA for high-performance compression operations, making it suitable for processing large files efficiently on systems with NVIDIA GPUs.

## General Setup

### Prerequisites
- **Operating System**: Windows (required for Windows Forms GUI)
- **.NET Runtime**: .NET 9.0 or later
- **Hardware**: NVIDIA GPU with CUDA support
- **Dependencies**: 
  - `nvcomp.dll` (NVIDIA nvCOMP library)
  - `cudart64_12.dll` (CUDA Runtime library)

### Development Environment
- **.NET SDK**: 9.0.x
- **Target Framework**: `net9.0-windows`
- **UI Framework**: Windows Forms
- **Language Features**: C# with nullable reference types enabled

### Build Process
The project supports both self-contained and framework-dependent builds:
- **Self-contained**: Includes .NET runtime, larger but portable
- **Framework-dependent**: Requires .NET 9.0 runtime, smaller and faster startup

## Repository Structure

```
GDeflate-Compressor/
├── .github/workflows/          # CI/CD automation
│   └── build.yml              # Build workflow for Windows x64
├── .openhands/microagents/    # AI agent documentation
│   └── repo.md               # This file
├── GDeflateGUI/              # Main application source
│   ├── Program.cs            # Application entry point
│   ├── MainForm.cs           # Main GUI form and event handlers
│   ├── MainForm.Designer.cs  # Windows Forms designer code
│   ├── GDeflateProcessor.cs  # Core compression/decompression logic
│   ├── NvCompApi.cs          # P/Invoke bindings for nvCOMP library
│   ├── CudaRuntimeApi.cs     # P/Invoke bindings for CUDA Runtime
│   └── GDeflateGUI.csproj    # Project configuration
├── LICENSE                   # GNU GPL v3 license
└── README.md                # User documentation and build instructions
```

### Key Components

#### Core Application (`GDeflateGUI/`)
- **Program.cs**: Standard Windows Forms application entry point
- **MainForm.cs**: Main UI with file selection, compression/decompression controls
- **GDeflateProcessor.cs**: Handles CUDA memory management and nvCOMP API calls
- **NvCompApi.cs**: Native interop layer for NVIDIA nvCOMP compression library
- **CudaRuntimeApi.cs**: Native interop layer for CUDA runtime operations

#### Features
- Batch file compression and decompression
- Drag-and-drop file selection
- Progress tracking and status updates
- GPU memory management with proper cleanup
- Error handling for CUDA and nvCOMP operations

## CI/CD Configuration

### GitHub Actions Workflow (`.github/workflows/build.yml`)
- **Triggers**: Push to `main`/`develop` branches, pull requests to `main`, manual dispatch
- **Runner**: `windows-latest`
- **Build Matrix**: Two parallel jobs
  1. **Self-contained build** (`GDeflate-Compressor-Windows-x64`)
     - Includes .NET runtime
     - Target: `win-x64`
     - Output: Portable executable bundle
  2. **Framework-dependent build** (`GDeflate-Compressor-Framework-Dependent`)
     - Requires .NET 9.0 runtime on target machine
     - Smaller deployment size
- **Artifacts**: 30-day retention with build metadata
- **Build Steps**:
  - Checkout code (`actions/checkout@v4`)
  - Setup .NET 9.0.x (`actions/setup-dotnet@v4`)
  - Restore dependencies
  - Build in Release configuration
  - Publish applications
  - Upload artifacts with build information

### Build Outputs
- Automated builds triggered by code changes
- Downloadable artifacts from GitHub Actions
- Build metadata including commit SHA, branch, and timestamp
- No linting, testing, or pre-commit hooks currently configured

## License
GNU General Public License v3.0 - Open source copyleft license ensuring derivative works remain open source.