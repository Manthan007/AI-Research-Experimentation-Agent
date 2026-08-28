import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "REA"

list_of_files = [
    ".github/workflows/.gitkeep",
    "config/config.yaml",
    "config/retrieval.yaml",
    "config/evaluation.yaml",
    "config/experiments.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/ingestion/__init__.py",
    f"src/{project_name}/chunking/__init__.py",
    f"src/{project_name}/embeddings/__init__.py",
    f"src/{project_name}/retrieval/__init__.py",
    f"src/{project_name}/agents/__init__.py",
    f"src/{project_name}/graph/__init__.py",
    f"src/{project_name}/tools/__init__.py",
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/experiments/__init__.py",
    f"src/{project_name}/report/__init__.py",
    "requirements.txt",
    'setup.py',
    'research/trials.ipynb',

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} is already exists")