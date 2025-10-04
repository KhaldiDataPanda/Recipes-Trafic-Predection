@echo off
echo Starting Recipe Traffic Classifier Application...
echo.
echo Starting FastAPI server in background...
start "FastAPI Server" cmd /k "python api.py"

timeout /t 3 /nobreak > nul

echo.
echo Starting Streamlit app...
streamlit run app.py

pause
