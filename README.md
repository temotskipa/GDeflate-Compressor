# GDeflate Compressor

A GPU-accelerated deflate compression application using NVIDIA's nvCOMP library, available in both GUI and console versions.

## Applications

### GDeflateGUI
A Windows Forms desktop application with an intuitive graphical interface for file compression and decompression operations.

### GDeflateConsole  
A cross-platform command-line application that provides the same compression functionality through a terminal interface.

## Building the Application

### Automated Build (GitHub Actions)

This repository includes a GitHub Actions workflow that automatically builds the application when code is pushed to the `main` or `develop` branches, or when a pull request is created.

The workflow creates two build variants:

1. **Self-contained executable** (`GDeflate-Compressor-Windows-x64`):
   - Includes the .NET runtime
   - Can run on Windows machines without .NET 9.0 installed
   - Larger file size but more portable

2. **Framework-dependent executable** (`GDeflate-Compressor-Framework-Dependent`):
   - Requires .NET 9.0 runtime to be installed on the target machine
   - Smaller file size
   - Faster startup time

### Manual Build

To build the applications manually:

1. Install [.NET 9.0 SDK](https://dotnet.microsoft.com/download/dotnet/9.0)
2. Clone this repository
3. Navigate to the project directory
4. Run the following commands:

#### Building the GUI Application

```bash
# Restore dependencies
dotnet restore GDeflateGUI/GDeflateGUI.csproj

# Build the application
dotnet build GDeflateGUI/GDeflateGUI.csproj --configuration Release

# Publish self-contained executable (Windows only)
dotnet publish GDeflateGUI/GDeflateGUI.csproj --configuration Release --output ./publish-gui --self-contained true --runtime win-x64

# Or publish framework-dependent executable (Windows only)
dotnet publish GDeflateGUI/GDeflateGUI.csproj --configuration Release --output ./publish-gui --self-contained false
```

#### Building the Console Application

```bash
# Build the console application (cross-platform)
dotnet build GDeflateConsole/GDeflateConsole.csproj --configuration Release

# Publish for current platform
dotnet publish GDeflateConsole/GDeflateConsole.csproj --configuration Release --output ./publish-console

# Or publish for specific platforms
dotnet publish GDeflateConsole/GDeflateConsole.csproj --configuration Release --output ./publish-console-win --runtime win-x64
dotnet publish GDeflateConsole/GDeflateConsole.csproj --configuration Release --output ./publish-console-linux --runtime linux-x64
dotnet publish GDeflateConsole/GDeflateConsole.csproj --configuration Release --output ./publish-console-mac --runtime osx-x64
```

### Downloading Built Executables

1. Go to the [Actions](../../actions) tab in this repository
2. Click on the latest successful build
3. Download the desired artifact:
   - `GDeflate-Compressor-Windows-x64` for the self-contained version
   - `GDeflate-Compressor-Framework-Dependent` for the framework-dependent version

## Usage

### GUI Application (GDeflateGUI)

1. Launch the application
2. Use "Add Files..." to select individual files for compression
3. Use "Add Folder..." to add all files from a directory
4. Click "Compress" to compress all added files (creates .gdef files)
5. Use "Decompress" to decompress .gdef files back to their original format

**Features:**
- Drag-and-drop file selection
- Batch compression/decompression
- Progress tracking
- Error handling with user-friendly messages
- Cross-platform compatibility improvements

### Console Application (GDeflateConsole)

The console application provides command-line access to compression functionality with **automatic GPU detection and fallback**:

```bash
# Compress a file
GDeflateConsole compress input.txt

# Decompress a file
GDeflateConsole decompress input.gdef

# List files in a directory
GDeflateConsole list [directory-path]

# Run built-in tests
GDeflateConsole test
```

**GPU Acceleration:**
- **Automatic Detection**: The application automatically detects CUDA and nvCOMP availability
- **GPU Mode**: Uses NVIDIA GPU acceleration when available (Windows with CUDA runtime and nvCOMP library)
- **Simulation Mode**: Falls back to simulation mode for testing on systems without GPU support
- **Cross-Platform**: Runs on Windows, Linux, and macOS (simulation mode on non-Windows platforms)

**Examples:**
```bash
# Compress a single file (GPU accelerated if available)
dotnet run --project GDeflateConsole compress document.pdf

# Decompress a file (GPU accelerated if available)
dotnet run --project GDeflateConsole decompress document.pdf.gdef

# List files in current directory
dotnet run --project GDeflateConsole list .

# Run comprehensive tests
dotnet run --project GDeflateConsole test

# List files in specific directory
dotnet run --project GDeflateConsole list /path/to/files
```

## Technical Implementation

### GPU Acceleration Architecture
The application uses a sophisticated GPU detection and fallback system:

1. **Runtime Detection**: Automatically detects CUDA runtime and nvCOMP library availability
2. **Graceful Fallback**: Seamlessly switches to simulation mode when GPU resources are unavailable
3. **Cross-Platform Support**: Maintains functionality across Windows, Linux, and macOS
4. **Memory Management**: Proper CUDA memory allocation and cleanup with comprehensive error handling

### Console Application Features
- **Dual-Mode Operation**: Real GPU acceleration when available, simulation mode otherwise
- **Built-in Testing**: Comprehensive test suite for validation
- **Performance Monitoring**: Timing and compression ratio reporting
- **Error Recovery**: Automatic cleanup of partial files on failure

## Recent Improvements

### Enhanced Error Handling
- Added comprehensive error handling to prevent crashes during file operations
- Improved user feedback with clear error messages and recovery suggestions
- Platform detection for better cross-platform compatibility

### Console Application GPU Integration
- **Real GPU Support**: Full integration with CUDA and nvCOMP libraries
- **Automatic Detection**: Runtime detection of GPU capabilities
- **Simulation Fallback**: Maintains functionality on systems without GPU support
- **Cross-platform Testing**: Built-in test suite works on all platforms

### Robustness Improvements
- Better handling of file selection dialog failures
- Graceful degradation when platform-specific features are unavailable
- Enhanced status reporting and user guidance
- Proper resource cleanup and memory management

## Requirements

### For GUI Application
- Windows operating system (Windows Forms requirement)
- .NET 9.0 runtime (for framework-dependent builds)
- NVIDIA GPU with CUDA support (for actual GPU acceleration)

### For Console Application
- Any operating system supported by .NET 9.0 (Windows, Linux, macOS)
- .NET 9.0 runtime
- NVIDIA GPU with CUDA support (for actual GPU acceleration, simulation mode available otherwise)

## License

This project is licensed under the terms specified in the LICENSE file.