from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from apis import v1, v2


app = FastAPI(title='YoloV3 Object Detection')

app.include_router(v1.app, prefix='/v1')
app.include_router(v2.app, prefix='/v2')

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root():
    return "For object detection: <br> \
            1. Use http://127.0.0.1:8000/v1 for Web UI access <br> \
            2. Use http://127.0.0.1:8000/v2 for CURL or Programmatic access <br><br>\
            You can also access API docs using http://127.0.0.1:8000/docs"

if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)