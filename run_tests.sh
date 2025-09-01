#!/bin/bash
export DOTNET_ROOT=$(pwd)/dotnet
export PATH=$PATH:$DOTNET_ROOT
$DOTNET_ROOT/dotnet run --project GDeflateConsole/GDeflateConsole.csproj -- test
