from fastapi import APIRouter, UploadFile, File, HTTPException

from detector import image_detection


app = APIRouter()

@app.get("/")
async def root():
    return "Go to http://127.0.0.1:8000/v2/detect to for CURL/Programmatic use."

@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.5):

    filename = file.filename
    fileExtension = filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not fileExtension:
        return HTTPException(status_code=415, detail="Unsupported file uploaded. App only supports jpg, jpeg and png formats.")

    contents = await file.read()
    results = image_detection(contents, confidence)

    if len(results) == 0:
        return "Failed to detect objects."

    return results