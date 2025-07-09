from fastapi import FastAPI

from src.models import ExplainRequest, Question, Option, ExplanationResponse

from .workflow import Workflow

app = FastAPI(
    title="PDF Question Extractor API",
    description="Extract multiple choice questions from PDF exam files",
    version="1.0.0"
)


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplainRequest):
    workflow = Workflow()
    state = await workflow.run(
        request.question,
        certification_id=request.filter.certification_id,
        tech=request.filter.tech,
        tech_id=request.filter.tech_id,
        training_slug=request.filter.training_slug
    )

    return ExplanationResponse(
        question=request.question,
        option_explanations=state.option_explanations,
        is_complete=state.is_complete,
        training_references=state.training_references
    )


@app.post("/test", response_model=ExplanationResponse)
async def test():
    """Test endpoint with sample MCQ question"""
    
    sample_question = Question(
        title="When is throttling more appropriate than debouncing?",
        description="",
        options=[
            Option(option="when you need to delay execution until user input stops", is_correct=False),
            Option(option="when you need regular updates at a fixed interval during continuous events", is_correct=True), 
            Option(option="when you want to prevent all rapid-fire events", is_correct=False),
            Option(option="when you want to cache function results", is_correct=False)
        ]
    )
    
    workflow = Workflow()
    state = await workflow.run(sample_question)
    
    return ExplanationResponse(
        question=sample_question,
        option_explanations=state.option_explanations,
        is_complete=state.is_complete,
        training_references=state.training_references
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF Question Extractor API is running"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Question Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "POST /explain": "Generate explanations for MCQ options",
            "GET /test": "Test endpoint with sample question",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 