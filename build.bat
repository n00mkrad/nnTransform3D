@echo off

if not defined VSCMD_VER (
    echo Initializing MS Build Tools...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo MS Build Tools environment already initialized.
)

echo Compiling: %~dp0build
cmake -S "%~dp0." -B "%~dp0build"

if %errorlevel% neq 0 (
    echo Aborting after running cmake -B - ERRORLEVEL %errorlevel%
    goto end
)

cmake --build "%~dp0build" --config Release -j 8

if %errorlevel% neq 0 (
    echo Finished with ERRORLEVEL %errorlevel%
) else (
    echo Copying ONNX Runtime DLLs and model weights...
    copy "%~dp0onnxruntime-win-x64-gpu\lib\*.dll" "%~dp0build\Release\"
    copy "%~dp0chroma_net.onnx" "%~dp0build\Release\"
    echo Done.
)

:end