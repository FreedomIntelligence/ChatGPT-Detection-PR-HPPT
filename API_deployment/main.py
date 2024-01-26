from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import numpy as np
from typing import List

class LogisticModel(torch.nn.Module):  
    def __init__(self,model):
        super().__init__()  
        self.softmax = torch.nn.Softmax(dim=1)  
        self.model = model
 
    def forward(self, inputs):
        logits = self.model(inputs["input_ids"].cuda(),
         attention_mask=inputs["attention_mask"].cuda()
        ).logits
        y_pred = self.softmax(logits)[:,1]
        return y_pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    return inputs


def get_prediction(model, input_ids, attention_mask):
    # Get the predicted label using the input_ids and attention_mask
    softmax = torch.nn.Softmax(dim=1)
    outputs = model(input_ids, attention_mask=attention_mask)
    score = softmax(outputs.logits)
    score = score.detach().cpu().numpy()[0]
    return[{"label":"Human","score":float(score[0])},{"label":"ChatGPT","score":float(score[1])}]
    '''
    if predicted_label == 0:
            return "Human"
    else:
        return "ChatGPT"
    '''
    

def get_PR(model, inputs):
    # Get the predicted label using the input_ids and attention_mask
    outputs = model(inputs)
    predicted_score = outputs.detach().cpu().numpy()
    return predicted_score

# 创建一个FastAPI应用
app = FastAPI()

# 在启动时加载模型和tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model_det = RobertaForSequenceClassification.from_pretrained("roberta-base")
model_PR = LogisticModel(RobertaForSequenceClassification.from_pretrained("roberta-base"))

# 创建一个用于接收用户请求的数据模型
class Item(BaseModel):
    texts: List[str]


@app.on_event("startup")
async def load_model():
    # 在启动时加载模型到显存
    model_det.to('cuda')
    model_PR.to('cuda')
    model_det.load_state_dict(torch.load("../Detecting-Generated-Abstract-main/best_model_paper.pt"))
    model_PR.load_state_dict(torch.load("../regress/last_model_reg_MSE.pt"))
    

@app.post("/predict")
async def predict(item: Item):
    # 接收用户的请求，将用户的文本送入模型，然后将结果返回给用户
    text_list = item.texts
    resuls=[]
    for text in text_list:
        inputs = preprocess_text(
                    tokenizer, text, max_length=512
                )
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        predicted_label = get_prediction(model_det, input_ids, attention_mask)
        resuls.append(predicted_label)
        #resuls.append({"label":predicted_label})
    return resuls

'''
if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model_det = RobertaForSequenceClassification.from_pretrained("roberta-base").cuda()
    text = "hhhh"
    inputs = preprocess_text(
                    tokenizer, text, max_length=512
                )
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    predicted_label = get_prediction(model_det, input_ids, attention_mask)
    print(predicted_label)
'''