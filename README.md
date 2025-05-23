# Generative AI Multi-Agent System on Databricks

This project demonstrates how to build, deploy, and evaluate generative AI multi-agent systems using the Databricks Data Intelligence Platform. 
The use case is to guide

## FEATURES
End-to-end workflow for GenAI app development: from proof-of-concept to production.
Integration with Databricks Mosaic AI for model serving, vector search, and agent orchestration.
Support for multiple GenAI architectural patterns: Prompt Engineering, Retrieval Augmented Generation (RAG), Fine-tuning, and Pretraining.
Example notebooks and templates for rapid prototyping.
MLflow integration for experiment tracking and agent evaluation.
Guidance on best practices for governance, monitoring, and continuous improvement.

## PREREQUISITES:

Access to a Databricks workspace. Databricks features: Mosaic AI, 

Python, Langchain, MLflow

## PROJECT STRUCTURE
src/
├── data/
├── vector_search/
│   ├── unstructured-data-pipeline - documentació # Vector search creation for the documentation sources. 
│   ├── unstructured-data-pipeline - tramits # Vector search creation for the tramit sources. 
├── architecture/
│   ├── langgraph-multiagent-genie-pat # Creation of the architecture
│   ├── agent. 
├── evaluation/
│   ├── run_evaluation.py        # Scripts to run evaluations (using MLflow, Mosaic AI, etc.)
├── models/
│   ├── 
├── utils/
├── config/
└── __init__.py
/docs - Documentation and architecture diagrams
/data - Sample datasets and data preparation scripts


## DEVELOPMENT WORKFLOW
1. Gather Requirements: Define the problem, objectives, and value proposition for your GenAI solution.

2. Design Architecture: Choose the appropriate agent pattern (Prompt, RAG, Fine-tuning, Pretraining).

3. Prepare Data: Clean, structure, and catalog your data using Unity Catalog.

4. Build Prototype: Use provided notebooks to create an initial agent or chain with minimal prompts and tools.

5. Deploy & Test: Serve models with Mosaic AI Model Serving and test using AI Playground or API endpoints.

6. Evaluate & Iterate: Log experiments with MLflow, collect feedback, and refine agent logic.

7. Productionize: Monitor performance, set up guardrails, and ensure compliance and governance.

## KEY TECHNOLOGIES
Databricks 
Unity Catalog (Data and model governance)
Mosaic AI (Model Serving, Vector Search)
MLflow (Experiment tracking and evaluation)
Open-source and external LLMs (e.g., Llama, Mistral, GPT-4, Claude)


