# Import required libraries
import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


def load_model(model_path):
    # Load the tokenizer and model from the "roberta-base" pre-trained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base").cuda()

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
        truncation=True,
    )

    # Get the input_ids and attention_mask tensors
    return inputs["input_ids"].cuda(), inputs["attention_mask"].cuda()


def get_prediction(model, input_ids, attention_mask):
    # Get the predicted label using the input_ids and attention_mask
    outputs = model(input_ids, attention_mask=attention_mask)
    predicted_label = np.argmax(outputs.logits.detach().cpu().numpy())
    return predicted_label


def main():
    # Load the fine-tuned model from the saved state dict
    model_path = "best_model.pt"
    tokenizer, model = load_model(model_path)

    # Get the test sentence from the file
    with open("text_test.txt", encoding="utf-8", mode="r") as fr:
        lines = fr.readlines()
        for line in lines:
            test_sentence = line.strip()
            # Preprocess the test sentence and get the predicted label
            input_ids, attention_mask = preprocess_text(
                tokenizer, test_sentence, max_length=512
            )
            predicted_label = get_prediction(model, input_ids, attention_mask)

            # Print the output based on the predicted label
            if predicted_label == 0:
                print("human")
            else:
                print("chatGPT")


if __name__ == "__main__":
    main()
