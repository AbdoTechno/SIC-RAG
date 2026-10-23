"""Main application entry point linking configuration and pipelines."""

import yaml
from rag_engine.pipeline.ingestion_pipeline import IngestionPipeline
from rag_engine.pipeline.retrieval_pipeline import RetrievalPipeline

def load_config(config_path: str = "rag_engine/configs/pipeline_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    """Application bootstrap routine."""
    config = load_config()
    print("Initializing RAG Engine with config version:", config.get("version"))
    
    # Initialize pipelines
    ingestion = IngestionPipeline(config)
    retrieval = RetrievalPipeline(config)
    
    print("RAG Engine successfully initialized and ready.")

if __name__ == "__main__":
    main()
