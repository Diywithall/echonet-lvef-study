@echo off
echo Rebuilding venv with Python 3.11...
cd /d c:\Users\sarda\Downloads\pocus\pocus\backend

REM Remove old broken venv
rmdir /s /q venv

REM Create new venv with Python 3.11
C:\Users\sarda\AppData\Local\Programs\Python\Python311\python.exe -m venv venv

echo Installing packages...
venv\Scripts\pip.exe install --upgrade pip
venv\Scripts\pip.exe install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip.exe install opencv-python-headless numpy scipy fastapi uvicorn python-multipart matplotlib pandas scikit-learn

echo Done.
venv\Scripts\python.exe --version
