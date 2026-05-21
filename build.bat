@echo off

echo --------------------------------------------

SET "BUILD_TOOLS_BAT=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%BUILD_TOOLS_BAT%" (
    echo Error: MS Build Tools not found at "%BUILD_TOOLS_BAT%"
    echo Please install Visual Studio 2022 Build Tools or update the path in this script.
    goto end
)

if not defined VSCMD_VER (
    echo Initializing MS Build Tools...
    call "%BUILD_TOOLS_BAT%" x64
) else (
    echo MS Build Tools environment already initialized.
)

echo Preparing build directory: %~dp0build...
cmake -S "%~dp0." -B "%~dp0build"

if %errorlevel% neq 0 (
    echo Aborting after running cmake -B - ERRORLEVEL %errorlevel%
    goto end
)

echo Building with %NUMBER_OF_PROCESSORS% parallel jobs...
cmake --build "%~dp0build" --config Release -j %NUMBER_OF_PROCESSORS%

if %errorlevel% neq 0 (
    echo Finished with ERRORLEVEL %errorlevel%
) else (
    echo Copying ONNX Runtime DLLs and model weights...
    copy "%~dp0onnxruntime-win-x64-gpu\lib\*.dll" "%~dp0build\Release\"
    copy "%~dp0chroma_net.onnx" "%~dp0build\Release\"
    echo Done.
)

:end
echo --------------------------------------------