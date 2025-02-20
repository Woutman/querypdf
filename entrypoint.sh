#!/bin/bash

PYTHON_SCRIPT="run_model.py"

cat <<EOL > $PYTHON_SCRIPT
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name_or_path = "Alibaba-NLP/gte-multilingual-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path).save_pretrained("models/alibaba")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    revision="815b4a86b71f0ecba053e5814a6c24aa7199301e",
    trust_remote_code=True,
    torch_dtype=torch.float16
).save_pretrained("models/alibaba")

print("Model and tokenizer loaded successfully.")
EOL

python $PYTHON_SCRIPT

streamlit run ./main/app.py --server.fileWatcherType=none