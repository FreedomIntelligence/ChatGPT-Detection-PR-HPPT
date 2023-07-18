# Import required libraries
import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from train import LogisticModel
import json


def load_model(model_path):
    # Load the tokenizer and model from the "roberta-base" pre-trained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = LogisticModel(RobertaForSequenceClassification.from_pretrained("roberta-base")).cuda()

    # Load the saved state dict of the fine-tuned model
    model.load_state_dict(torch.load(model_path))

    return tokenizer, model


def preprocess_text(tokenizer, input_text, max_length):
    # Tokenize the input text using the tokenizer
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    )

    
    return inputs


def get_prediction(model, inputs):
    # Get the predicted label using the input_ids and attention_mask
    outputs = model(inputs)
    predicted_score = outputs.detach().cpu().numpy()
    return predicted_score


def test(model_path,test_data_path):
    # Load the fine-tuned model from the saved state dict
    tokenizer, model = load_model(model_path)
    model.eval()

    # Get the test sentence from the file
    with open(test_data_path, encoding="utf-8", mode="r") as fr:
        lines = fr.readlines()
        for line in lines:
            test_sentence = line.strip()
            # Preprocess the test sentence and get the predicted label
            inputs = preprocess_text(
                tokenizer, test_sentence, max_length=512
            )
            predicted_score = float(get_prediction(model, inputs)[0])
            print(predicted_score)
    
    
        



if __name__ == "__main__":
    test("last_model_reg_MSE.pt","test.txt")