import json
from tqdm import tqdm
import torch
from torch import nn, optim
import re
import datasets
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split

# Path to file of data
data_path = "./data/dataset.json"
data_path_open = "./data/open_source_dataset.json"

## LOADING DATA
with open(data_path, "r") as fichier1:
    data = json.load(fichier1)

with open(data_path_open, "r") as fichier2:
    data_open = json.load(fichier2)

# Extract variables from open source data
data_open_extracted = {}

# For each element in open data
for key, value in data_open.items():
    if "uid" in value and "reference_summary" in value and "original_text" in value:
        data_open_extracted[key] = {
            "uid": value["uid"],
            "reference_summary": value["reference_summary"],
            "original_text": value["original_text"]
        }

## DATA CONCATENATION
# Copy of data to avoid changing source
data_concatenee = data.copy()
data_concatenee.update(data_open_extracted)

# Path to created file
chemin_fichier_concatene = "./output/data_concatenee.json"

# Writing data to the new created file
with open(chemin_fichier_concatene, "w") as fichier_concatene:
    json.dump(data_concatenee, fichier_concatene)

print("Concaténation terminée. Données enregistrées dans", chemin_fichier_concatene)

## SPLIT AND TRAIN TEST
# Convert concatenated_data into tuples
data_list = list(data_concatenee.items())

# Split into training and test
train_data_list, test_data_list = train_test_split(data_list, test_size=0.2, random_state=42)

# Convert into dictionnaries
train_data = dict(train_data_list)
test_data = dict(test_data_list)

# Path to new created files
chemin_fichier_train = "./output/train_data.json"
chemin_fichier_test = "./output/test_data.json"

#  Writing data to the new created file
with open(chemin_fichier_train, "w") as fichier_train:
    json.dump(train_data, fichier_train)

with open(chemin_fichier_test, "w") as fichier_test:
    json.dump(test_data, fichier_test)

print("Données d'entraînement enregistrées dans", chemin_fichier_train)
print("Données de test enregistrées dans", chemin_fichier_test)

## NETTOYAGE DU DATASET

# Cleaning text function
def clean_text(text, remove_stopwords = True):
  # Delete useless ponctuation
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  text = re.sub(r'https?:\/\/.[\r\n]', '', text, flags=re.MULTILINE)
  text = re.sub(r'\<a href', ' ', text)
  text = re.sub(r'&amp;', '', text) 
  text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
  text = re.sub(r'<br />', ' ', text)
  text = re.sub(r'\'', ' ', text)  
  # Convert in lowercase
  text = text.lower() 
  return text

# Apply function to train_data
cleaned_data = {}
for key, item in train_data.items():
  original_text = item["original_text"]
  reference_summary = item["reference_summary"]

  # Clean original_text and summary
  cleaned_original_text = clean_text(original_text)
  cleaned_reference_summary = clean_text(reference_summary)

  # Add data to new dict
  cleaned_data[key] = {
      "original_text": cleaned_original_text,
      "reference_summary": cleaned_reference_summary,
      "uid": item["uid"]
    }

# Save dataset
with open("./output/data_concatenee.json", "w", encoding="utf-8") as f:
  json.dump(cleaned_data, f, ensure_ascii = False, indent = 4)

# Finalization
print("Nettoyage terminé. Le nouveau dataset nettoyé a été sauvegardé sous le nom 'cleaned_data.json'.")

## TRAINING MODEL
# Charge BART model and its tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Define Class Dataset for training data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())  # Convertir les valeurs du dictionnaire en liste

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        original_text = sample["original_text"]
        reference_summary = sample["reference_summary"]
        return {"original_text": original_text, "reference_summary": reference_summary}

# Intialize dataloader
train_dataset = CustomDataset(cleaned_data)
train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training the model
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        input_texts = []
        target_texts = []

        for item in batch["original_text"]:
            input_texts.append(item)

        for item in batch["reference_summary"]:
            target_texts.append(item)

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=150).input_ids

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# Save the model
model.save_pretrained("./output/model")
tokenizer.save_pretrained("./output/model")

print("Entraînement terminé. Modèle sauvegardé.")
