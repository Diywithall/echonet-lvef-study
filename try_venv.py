import subprocess, sys
# Try to run the venv python directly via its activation
result = subprocess.run(
    ['c:\\Users\\sarda\\Downloads\\pocus\\pocus\\backend\\venv\\Scripts\\python.exe', '--version'],
    capture_output=True, text=True, shell=True
)
print("returncode:", result.returncode)
print("stdout:", result.stdout)
print("stderr:", result.stderr)
