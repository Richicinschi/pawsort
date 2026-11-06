# PawSort
 
Upload a camera trap photo and find out if it is empty or which animal it shows, with a "needs review" flag when the model is not sure.

PawSort is built from three servers: the interface (port 8000) talks to the orchestrator (port 8001, SQLite), and the AI server takes jobs from the orchestrator. The AI server works in two steps: MegaDetector finds the animal, then our MobileNet names it. The interface never talks to the AI server directly.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
 