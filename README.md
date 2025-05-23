# Generative AI Multi-Agent System on Databricks

This project demonstrates how to build, deploy, and evaluate generative AI multi-agent systems using the Databricks Data Intelligence Platform. 
The use case is to guide new entrepreneurs to create their startup through oficial documents and procedures. 

## PREREQUISITES:

Access to a Databricks workspace. Databricks features: Mosaic AI, 
Python, Langchain, MLflow.

## PROJECT STRUCTURE
```
src/
├── data/
│   ├── import-dades-obertes-api.py 
│   ├── import-csv-into-volume.py 
│   ├── web-scrapping.py
├── vector_search/
│   ├── vector-search-documentation.py # Vector search creation for the documentation sources. (private)
│   ├── vector-search-tramits.py # Vector search creation for the tramit sources. (private)
├── architecture/
│   ├── langgraph-multiagent-genie-pat # Development and deploy of the model (private)
│   ├── agent.py # Architecture multi-agent (public)
├── evaluation/
│   ├── run_evaluation.py        # Scripts to run evaluations (using MLflow, Mosaic AI, etc.) (private)
├── models/
│   ├── 
├── utils/
├── config/
└── __init__.py
/docs - Documentation and architecture diagrams
/data - Sample datasets and data preparation scripts
```

## DEVELOPMENT WORKFLOW
1. Load the data to Unity Catalog (volumes and tables) and preprocess.
2. Develope vector search index and genie.
3. Design and develope the model ( a multi-agent system architecture).
4. Deploy & Test: Serve models with Mosaic AI Model Serving and test using AI Playground or API endpoints.
5. Evaluate & Iterate: Log experiments with MLflow, collect feedback, and refine agent logic.
7. Productionize: Monitor performance, set up guardrails, and ensure compliance and governance.

## KEY TECHNOLOGIES
Databricks 
Unity Catalog (Data and model governance)
Mosaic AI (Model Serving, Vector Search)
MLflow (Experiment tracking and evaluation)
Open-source and external LLMs (e.g., Llama, Mistral, GPT-4, Claude)


