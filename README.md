# GDeflate Compressor

A Windows Forms application for GPU-accelerated deflate compression using NVIDIA's nvCOMP library.

## Building the Application

### Automated Build (GitHub Actions)

This repository includes a GitHub Actions workflow that automatically builds the application when code is pushed to the `main` or `develop` branches, or when a pull request is created.

The workflow creates two build variants:

1. **Self-contained executable** (`GDeflate-Compressor-Windows-x64`):
   - Includes the .NET runtime
   - Can run on Windows machines without .NET 6.0 installed
   - Larger file size but more portable

2. **Framework-dependent executable** (`GDeflate-Compressor-Framework-Dependent`):
   - Requires .NET 6.0 runtime to be installed on the target machine
   - Smaller file size
   - Faster startup time

### Manual Build

To build the application manually:

1. Install [.NET 6.0 SDK](https://dotnet.microsoft.com/download/dotnet/6.0)
2. Clone this repository
3. Navigate to the project directory
4. Run the following commands:

```bash
# Restore dependencies
dotnet restore GDeflateGUI/GDeflateGUI.csproj

# Build the application
dotnet build GDeflateGUI/GDeflateGUI.csproj --configuration Release

# Publish self-contained executable
dotnet publish GDeflateGUI/GDeflateGUI.csproj --configuration Release --output ./publish --self-contained true --runtime win-x64

# Or publish framework-dependent executable
dotnet publish GDeflateGUI/GDeflateGUI.csproj --configuration Release --output ./publish --self-contained false
```

### Downloading Built Executables

1. Go to the [Actions](../../actions) tab in this repository
2. Click on the latest successful build
3. Download the desired artifact:
   - `GDeflate-Compressor-Windows-x64` for the self-contained version
   - `GDeflate-Compressor-Framework-Dependent` for the framework-dependent version

## Requirements

- Windows operating system
- .NET 6.0 runtime (for framework-dependent builds)
- NVIDIA GPU with CUDA support (for GPU acceleration)

## License

This project is licensed under the terms specified in the LICENSE file.