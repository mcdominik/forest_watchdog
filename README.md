# forest_watchdog

PyTorch model handled with FastAPI+uvicorn.


### API:

- / -> returns: app status

- /predict -> upload file
[takes: images] ->
returns: prediction

- /getMyDevice ->
returns: current backend mode (cpu or gpu)
