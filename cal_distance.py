import textdistance
import torch
from transformers import AutoTokenizer, AutoModel


class SimCalculator:
    def __init__(self, model_name="sci_bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def bert_similarity(self, text1, text2):
        # 对输入文本进行分词
        inputs1 = self.tokenizer(text1, return_tensors="pt")
        inputs2 = self.tokenizer(text2, return_tensors="pt")
        inputs1 = inputs1.to(self.device)
        inputs2 = inputs2.to(self.device)        # 计算文本的 BERT 词向量
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)

        # 提取[CLS]标记的向量作为文本表示
        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu()
        return cos_sim.item()

    def cal_similarity(self, text1, text2):
        split_1 = text1.split(" ")
        split_2 = text2.split(" ")
        levenshtein_distance = textdistance.levenshtein.normalized_distance(split_1, split_2)
        jaccard_distance = textdistance.jaccard.normalized_distance(split_1, split_2)
        sem_similarity = self.bert_similarity(text1, text2)
        return_dict = {
            "levenshtein_distance": round(levenshtein_distance, 4),
            "jaccard_distance": round(jaccard_distance, 4),
            "sem_similarity": round(sem_similarity, 4)
        }
        return return_dict


if __name__ == '__main__':
    text1 = "This paper focuses on subword-based Neural Machine Translation (NMT). We hypothesize that in the NMT model, the appropriate subword units for the following three modules (layers) can differ: (1) the encoder embedding layer, (2) the decoder embedding layer, and (3) the decoder output layer. We find the subword based on Sennrich et al. (2016) has a feature that a large vocabulary is a superset of a small vocabulary and modify the NMT model enables the incorporation of several different subword units in a single embedding layer. We refer these small subword features as hierarchical subword features. To empirically investigate our assumption, we compare the performance of several different subword units and hierarchical subword features for both the encoder and decoder embedding layers. We confirmed that incorporating hierarchical subword features in the encoder consistently improves BLEU scores on the IWSLT evaluation datasets."
    text2 = "This paper delves into the topic of subword-based Neural Machine Translation (NMT). Our hypothesis is that the appropriate subword units for the following three modules (layers) in the NMT model can vary: (1) the encoder embedding layer, (2) the decoder embedding layer, and (3) the decoder output layer. We adopt the subword approach proposed by Sennrich et al. (2016), which has the advantage of a large vocabulary being a superset of a smaller vocabulary. This enables us to modify the NMT model to incorporate multiple subword units in a single embedding layer, referred to as hierarchical subword features. To validate our hypothesis, we compare the performance of various subword units and hierarchical subword features for both the encoder and decoder embedding layers. Our results show that incorporating hierarchical subword features in the encoder consistently improves BLEU scores on the IWSLT evaluation datasets."
    simCalculator = SimCalculator("./sci_bert")
    return_dict = simCalculator.cal_similarity(text1, text2)
    print(return_dict)
    text3=text2
    return_dict = simCalculator.cal_similarity(text3, text2)
    print(return_dict)
