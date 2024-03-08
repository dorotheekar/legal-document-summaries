""" Generate summaries from trained model onto test_set"""

import json
from transfo    rmers import BartForConditionalGeneration, BartTokenizer

# Choose either test_set or test_data
data_input = input('Do you want to generate summaries for test_set or test_data ? Please enter either test_set or test_data: ')

# Load the file to predict
if data_input == 'test_set':
    data_name = 'data/test_set'

if data_input == 'test_data':
    data_name = 'output/test_data'

data_path = f"./{data_name}.json"
with open(data_path, "r") as data:
    data = json.load(data)

# Load the model and its tokenizer  
model_path = "./output/model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Function that uses the model to generate summaries
def generate_summary(original_text, uid):
    input_text = original_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=296, # total length of summary
        num_beams=8, # number of beams
        length_penalty=2.0, # apply a penalty to the length
        temperature=0.8, # enable 'creativity'
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"generated_summary": summary, "uid": uid}

# Apply the function every element of on data
generated_summaries = {}

for key, value in data.items():
    original_text = value["original_text"]
    uid = key
    summary_info = generate_summary(original_text, uid)
    generated_summaries[key] = {"original_text": original_text, **summary_info}

# Enregistrer les résumés générés dans un fichier JSON
output_json_path = f"./output/generated_summaries_{data_input}.json"
with open(output_json_path, "w+") as json_file:
    json.dump(generated_summaries, json_file, indent=2)

print(f"Résumés générés avec succès et enregistrés dans {output_json_path}")