# Restore pre-task Windows Balanced plan and GPU automatic clocks (administrator required).
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e
nvidia-smi -rgc
nvidia-smi -rmc
