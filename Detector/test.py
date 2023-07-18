# Import required libraries
import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn import metrics
from sklearn.metrics import classification_report
import json


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
    logits = outputs.logits
    score = torch.sigmoid(logits).detach().cpu().numpy()
    predicted_label = np.argmax(logits.detach().cpu().numpy())
    return predicted_label,float(score[0][1])


def test(model_path,test_data_path):
    predictions = []
    true_labels = []
    score = []
    error_samples = []
    # Load the fine-tuned model from the saved state dict
    tokenizer, model = load_model(model_path)

    # Get the test sentence from the file
    with open(test_data_path, encoding="utf-8", mode="r") as fp:
        info = json.load(fp)
        for data in info:
            test_sentence = data["text"]
            label = data["fake"]
            # Preprocess the test sentence and get the predicted label
            input_ids, attention_mask = preprocess_text(
                tokenizer, test_sentence, max_length=512
            )
            predicted_label,logits = get_prediction(model, input_ids, attention_mask)
            true_labels.append(label)
            predictions.append(predicted_label)
            score.append(logits)
            if predicted_label != label:
                error_samples.append(data)
    fp.close()
    report = classification_report(true_labels, predictions, digits=4)
    auc = metrics.roc_auc_score(true_labels,score)
    print(report)
    print("AUC:",auc)
        



if __name__ == "__main__":
    test(model_path=YOUR_MODEL_PATH,test_data_path=YOUR_TESTSET)