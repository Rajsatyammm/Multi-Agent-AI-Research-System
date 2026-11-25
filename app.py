from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from pipeline import run_research_pipeline

app = FastAPI(title="Multi-Agent AI Research System")


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, topic: str) -> str:
        topic = topic.strip()
        if not topic:
            raise ValueError("Topic must not be blank")
        return topic


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def home() -> dict:
    return {
        "status": "ok",
        "statusCode": 200,
        "message": "Welcome to AI Research Agent"
    }

@app.post("/research")
def research(request: ResearchRequest) -> dict:
    return run_research_pipeline(request.topic)
