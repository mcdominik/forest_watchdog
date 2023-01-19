from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from src.eval import ModifiedResNet18
from src.utils import my_device


app = FastAPI()
my_net = ModifiedResNet18("models/working.pth")

origins = [
    "https://mcdominik.github.io/forest_watchdog_front/",
    "http://localhost",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/getMyDevice")
def read_device():
    return {"mode": str(my_device)}


@app.post("/predict")
async def create_upload_file(file: UploadFile):
    image = await file.read()
    result = my_net.predict(image)
    return {"state": result}