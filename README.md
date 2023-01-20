# forest_watchdog

PyTorch model handled with FastAPI+uvicorn.


### API:

- / -> returns: app status

- /predict -> upload file
[takes: images] ->
returns: prediction

- /getMyDevice ->
returns: current backend mode (cpu or gpu)

## Web interface 

https://mcdominik.github.io/forest_watchdog_front/

front repository -> https://github.com/mcdominik/forest_watchdog_front
