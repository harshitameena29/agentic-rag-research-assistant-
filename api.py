import os
import shutil
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from graph import build_graph
from retriever import build_retriever

app = FastAPI(
    title="Agentic RAG Research Assistant",
    description="Upload a PDF and ask questions using Agentic RAG.",
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/research")
async def research(
    question: str = Form(...),
    file: UploadFile = File(...)
):

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(
                file.file,
                temp_file
            )

        retriever = build_retriever(
            temp_path
        )

        research_graph = build_graph(
            retriever=retriever
        )

        result = research_graph.invoke(
            {
                "question": question,
                "sub_questions": [],
                "current_index": 0,
                "findings": [],
                "final_report": "",
                "retriever": retriever,
            }
        )

        return {
            "report": result.get(
                "final_report",
                ""
            ),
            "sub_questions": result.get(
                "sub_questions",
                []
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass