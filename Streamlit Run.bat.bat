@echo off
setlocal enabledelayedexpansion


:task1
cls
set "extension1=.py"
set "counter1=1"
echo Searching for files with extension %extension1% in the current directory...

set "selectedFile="

for %%F in (*%extension1%) do (
    echo !counter1! - %%~nF.py
    set "file[!counter1!]=%%F"
    set /a "counter1+=1"
)

set /p "choice1=Enter the number of the file: "


if not defined file[%choice1%] (
    echo Invalid choice. Please enter a valid number.
    pause
    goto task1
)

set "fullFileName1=!file[%choice1%]!"

if exist "!fullFileName1!" (
    echo Selected file: !fullFileName1!
    streamlit run "%fullFileName1%"

    pause
    goto mainMenu

) else (
    echo Invalid choice. Selected file does not exist.
    pause
    goto task1
)

pause
goto :eof
