# ChatGPT-Detection-PR-HPPT 
This our dataset and code for the paper: [Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect ChatGPT-Generated Text](https://arxiv.org/pdf/2307.11380.pdf)
![The overall design of our method.](image.png)


# Guideline

The detector is a Roberta for classification model with labels (0: human, 1:ChatGPT-involved).

If you want to train it, follow these steps:

1. install the environment

```bash
pip install -r requirements.txt
```

2. train a detector
```bash
cd Detector
```
```bash
python train.py
```

It is also all right for you to change some setting in the code.

3. get the detector

The ```best_model.pt``` is the trained detector.

You can test the custom sample in text_test.txt (only three examples in it):

```bash
python inference.py
```

If you do not want to train the model, we provide our trained detector on HPPT: [Trained Detector](https://drive.google.com/file/d/10qTNMj4Fo1GwNXhWtlM5RZK5VXsgOsoD/view?usp=drive_link)

4. train a model to get polish ratio
```bash
cd ../PR_reg
```

```bash
python train.py
```

We also provide the trained PR model: [Trained PR model](https://drive.google.com/file/d/1WquVC6ei-gkNE_oHm9W6N5iR8gu5XjLB/view?usp=drive_link)

# Citation
You are welcome to use our dataset and models. 
For citation following BibTex entry: 
```bash
@misc{yang2023chatgpt,
      title={Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect ChatGPT-Generated Text}, 
      author={Lingyi Yang and Feng Jiang and Haizhou Li},
      year={2023},
      eprint={2307.11380},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
