# PawSort
 
Upload a pet photo and get "cat" or "dog" back, with a "needs review" flag when the model is not sure.

PawSort is built from three servers: the interface (port 8000) talks to the orchestrator (port 8001, SQLite), and the AI server takes jobs from the orchestrator. The interface never talks to the AI server directly.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
 