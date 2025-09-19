@echo off
echo Building Consciousness Emulator v1.1 for Windows...
echo.

REM Check if gcc is available
gcc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: gcc not found. Please install MinGW64 or add it to PATH.
    pause
    exit /b 1
)

echo Creating build directories...
if not exist build mkdir build
if not exist build\kernel mkdir build\kernel
if not exist build\wm mkdir build\wm
if not exist build\utils mkdir build\utils
if not exist build\demos mkdir build\demos
if not exist bin mkdir bin

echo.
echo Compiling core library...
gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -I./include -I./src -c src/kernel/kernel.c -o build/kernel/kernel.o
if errorlevel 1 goto error

gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -I./include -I./src -c src/kernel/message_queue.c -o build/kernel/message_queue.o
if errorlevel 1 goto error

gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -I./include -I./src -c src/wm/working_memory.c -o build/wm/working_memory.o
if errorlevel 1 goto error

gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -I./include -I./src -c src/utils/math_utils.c -o build/utils/math_utils.o
if errorlevel 1 goto error

gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -I./include -I./src -c src/consciousness.c -o build/consciousness.o
if errorlevel 1 goto error

echo Creating static library...
ar rcs build/libconsciousness.a build/kernel/kernel.o build/kernel/message_queue.o build/wm/working_memory.o build/utils/math_utils.o build/consciousness.o
if errorlevel 1 goto error

echo.
echo Compiling demo...
gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -I./include -I./src -c src/demos/simple_demo.c -o build/demos/simple_demo.o
if errorlevel 1 goto error

echo Linking demo executable...
gcc -std=c11 -Wall -Wextra -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -o bin/ce_demo.exe build/demos/simple_demo.o build/libconsciousness.a -lm -lpthread -ldl
if errorlevel 1 goto error

echo.
echo Build successful!
echo.
echo To run the demo:
echo   bin\ce_demo.exe
echo.
echo To clean build files:
echo   del /s /q build
echo   del /s /q bin
echo.
pause
exit /b 0

:error
echo.
echo Build failed! Check the error messages above.
pause
exit /b 1
