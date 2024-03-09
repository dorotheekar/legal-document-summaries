## Legal Text Summarization Model Training

### Overview

> This project focuses on text summarization of legal documents using the BART (Bidirectional and Auto-Regressive Transformers) model. The goal is to generate concise summaries from given input texts.
<!-- toc -->

### Table of contents

- [Key Components](#key-components)
- [Projet structure](#projet-structure)
- [Getting started](#getting-started)

### Key Components

1. **Datasets:**
   - ```dataset.json``` and ```open_source_dataset.json``` de forme 
   - Each data point includes original text, reference summary, and a unique identifier (uid).

2. **Model:**
   - Used the ```facebook/bart-base``` pre-trained BART model.
   - Approximately 139 million parameters.

3. **Training Process:**
   - Implemented training over three epochs.
   - AdamW optimizer with a learning rate of 1e-5.
   - CrossEntropyLoss used as the loss function.

4. **Inference:**
   - Summarization performed using the trained model.
   - Inference time details not provided.

### Project Structure
```
/ code_source
│ infer_summaries.py
│ README.md
│ requirements.txt
│ train_summarization_model.py
│ 
└───data
│ dataset.json
│ open_source_dataset.json
│ test_set.json
│ 
└───output
│ │
│ └───model 
│ data_concatenee.json
│ generated_summaries_test_data.json
│ generated_summaries_test_set.json
│ test_data.json
└ train_data.json
```

### Getting Started

1. Install dependencies : The model was developed and tested on Python 3.10.11, you can install all the dependencies using : 
```
$ pip install -r requirements.txt
```

2. Usage for model training  : This script refers to the model training process. It can be launch using :
```
$ python train_summarization_model.py
```

3. Generate summaries : This script will automatically generate summaries with the model by using :
```
$ python infer_summaries.py
```
