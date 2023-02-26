from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from detector import image_detection, encode_img_to_base64


app = APIRouter()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=RedirectResponse)
async def root(request: Request):
    return f"{request.url}detect"
    # return "Go to http://127.0.0.1:8000/v1/detect to use Web UI"

@app.get("/detect")
async def home(request: Request):
    return templates.TemplateResponse("home.html", context={"request": request})

@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...), confidence: float = 0.5):

    filename = file.filename
    fileExtension = filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not fileExtension:
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "error": True,
                "file": "Unsupported file uploaded. App only supports jpg, jpeg and png formats." 
            },
            status_code=415
        )

    contents = await file.read()
    labelled_img = image_detection(contents, confidence, True)
    pred_as_text = encode_img_to_base64(labelled_img).decode("utf-8")

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "error": False,
            "predicted_img": pred_as_text
        }
    )