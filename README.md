# Generative AI Multi-Agent System on Databricks

This project demonstrates how to build, deploy, and evaluate generative AI multi-agent systems using the Databricks Data Intelligence Platform. 

 **Use Case:**  
Guide new entrepreneurs through the process of starting a business using official documents and procedures from trusted sources (RAG + agents).

Built with Databricks Mosaic AI, LangChain, MLflow, and LLMs (Llama and Claude)
## PREREQUISITES:

Access to a Databricks workspace. Databricks features: Mosaic AI, 
Python, Langchain, MLflow.

## PROJECT STRUCTURE
```
src/
├── data/
│   ├── import-dades-obertes-api.py (private)
│   ├── import-csv-into-volume.py (private)
│   ├── web-scrapping.py (private)
├── vector_search/
│   ├── vector-search-documentation.py # Vector search creation for the documentation sources. (private)
│   ├── vector-search-tramits.py # Vector search creation for the tramit sources. (private)
├── architecture/
│   ├── langgraph-multiagent-genie-pat # Development and deploy of the model (private)
│   ├── agent.py # Architecture multi-agent (public)
├── evaluation/
│   ├── run_evaluation.py        # Scripts to run evaluations (using MLflow, Mosaic AI) (private)
├── models/
│   ├── (private)
├── utils/
├── config/
└── __init__.py

data/
│   ├── corpus-documents/
│   ├── datasets/
│   ├── evaluation-datasets/

```

## DEVELOPMENT WORKFLOW
1. Load the data to Unity Catalog (volumes and tables) and preprocess.
2. Develope vector search index and genie.
3. Design and develope the model ( a multi-agent system architecture).
4. Deploy & Test: Serve models with Mosaic AI Model Serving and test using AI Playground or API endpoints.
5. Evaluate & Iterate: Log experiments with MLflow, collect feedback, and refine agent logic.
7. Productionize: Monitor performance, set up guardrails, and ensure compliance and governance.

## KEY TECHNOLOGIES
- Databricks
    - Unity Catalog (Data and model governance)
    - Mosaic AI (Model Serving, Vector Search, Agent Evaluation)
    - Genie
- MLflow (Experiment tracking and evaluation)
- External LLMs ( Llama, Claude)

## LangGraph architecture
![Architecture Diagram](./media/langgraph.png)

## DEMO GIF

![Demo](./media/demo-1.gif)

