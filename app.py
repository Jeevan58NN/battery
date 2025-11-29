from custom_pinn import PINN, ResidualBlock
from fastapi import FastAPI, Header, HTTPException
import tensorflow as tf
import joblib
import numpy as np
import json
import os
import logging
import uvicorn
from rag import RAG
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Battery AI (PINN + RAG)")

API_KEY = os.getenv("API_KEY")

model = None
meta = None
templates = None


@app.on_event("startup")
async def startup_event():
    global model, meta, templates

    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Files: {os.listdir('.')}")

    # ---- Load Model ----
    try:
        custom_objects = {"PINN": PINN, "ResidualBlock": ResidualBlock}
        model = tf.keras.models.load_model(
            "pinn_inference.keras",
            compile=False,
            custom_objects=custom_objects
        )
        logger.info("✓ PINN model loaded")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

    # ---- Metadata ----
    try:
        meta = joblib.load("pinn_meta_full.pkl")
        logger.info("✓ Metadata loaded")
    except Exception as e:
        logger.error(f"❌ Error loading metadata: {e}")
        raise

    # ---- Templates ----
    try:
        with open("material_templates.json", "r") as f:
            templates = json.load(f)
        logger.info("✓ Templates loaded")
    except Exception as e:
        logger.error(f"❌ Error loading templates: {e}")

    # ---- RAG Index ----
    try:
        docs_path = Path(__file__).parent / "docs"
        logger.info(f"RAG docs folder: {docs_path}")

        RAG.load_index()

        if RAG.vectorizer is None:
            logger.info("No index found → building new RAG index...")
            RAG.build_index_from_folder(str(docs_path))

        logger.info(f"✓ RAG Ready. Docs: {len(RAG.doc_ids)}")
    except Exception as e:
        logger.warning(f"⚠️ RAG start failed: {e}")


# ------------------ RAG Endpoints ------------------

@app.post("/rag/reindex")
def rag_reindex():
    try:
        RAG.build_index_from_folder("docs")
        return {"status": "ok", "n_docs": len(RAG.doc_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/retrieve")
def rag_retrieve(data: dict):
    q = data.get("query", "")
    top_k = int(data.get("top_k", 5))

    if not q:
        raise HTTPException(status_code=400, detail="query required")

    hits = RAG.retrieve(q, top_k)
    out = [{"doc_id": d, "excerpt": t, "score": s} for (d, t, s) in hits]
    return {"query": q, "hits": out}


@app.post("/rag/answer")
def rag_answer(data: dict):
    q = data.get("query", "")
    if not q:
        raise HTTPException(status_code=400, detail="query required")

    ctx = RAG.get_context(q, top_k=3)
    return {"query": q, "answer": None, "context": ctx}


# ------------------ ROOT ------------------

@app.get("/")
def root():
    return {"status": "running", "api": "Battery AI"}


# ------------------ PREDICT ------------------

@app.post("/predict")
def predict(features: dict, x_api_key: str = Header(None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")

    if model is None or meta is None:
        raise HTTPException(503, "Model not loaded")

    try:
        x = []
        missing = []

        for name in meta["feature_names"]:
            if name not in features:
                missing.append(name)
            else:
                x.append(features[name])

        if missing:
            raise HTTPException(400, f"Missing: {missing}")

        x = np.array(x).reshape(1, -1)
        y = model.predict(x, verbose=0)

        return {"output": float(y[0][0]), "status": "success"}

    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


# ------------------ INFO ------------------

@app.get("/info")
def info():
    return {
        "features": meta.get("feature_names", []),
        "num": len(meta.get("feature_names", [])),
        "templates": templates is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
