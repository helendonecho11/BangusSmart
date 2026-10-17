"""Run with: python -m uvicorn api:app --host 127.0.0.1 --port 8000."""
from typing import Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator
from bangussmart_ml import advise
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dashboard import router
from api_media import router as media_router

app = FastAPI(title="BangusSmart synthetic prototype")
app.include_router(router)
app.include_router(media_router)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def dashboard_page():
    return FileResponse(STATIC / "index.html")
Finite = Annotated[float, Field(allow_inf_nan=False)]


class AdviceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temp: Finite
    ph: Annotated[Finite, Field(ge=0, le=14)]
    do: Annotated[Finite, Field(ge=0)]
    hour: int = Field(ge=0, le=23)
    do_history: list[Annotated[Finite, Field(ge=0)]]
    day: int = Field(ge=0)
    mean_temp: Finite
    fish_count: int = Field(ge=0)

    @field_validator("do_history")
    @classmethod
    def history_length(cls, values):
        if len(values) not in (3, 12):
            raise ValueError("Provide 3 or 12 hourly DO readings, oldest first")
        return values


@app.post("/advise")
def post_advise(request: AdviceRequest):
    try:
        return advise(**request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(503, "Train models with python bangussmart_ml.py all first") from exc
