import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from eval import ModifiedResNet18
from utils import my_device


app = FastAPI()
my_net = ModifiedResNet18("models/working.pth")

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "https://mcdominik.github.io/forest_watchdog_front/",
    "http://localhost",
    "https://mcdominik.github.io",
    "http://mcdominik.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://127.0.0.1:5500",
    "https://127.0.0.1:5501",
    "http://127.0.0.1:5501",


]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"AppStatus": "Running"}


@app.get("/getMyDevice")
def read_device():
    return {"mode": str(my_device)}


@app.post("/predict")
async def create_upload_file(file: UploadFile):
    image = await file.read()
    result = my_net.predict(image)
    return {"state": result}
