import uvicorn
import os
import sys

# Ensure the backend directory is in the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Ensure current directory is 'backend'
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the app
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
