import requests
import json

# 定义FastAPI服务的地址和端口
API_URL = "http://localhost:8000"

def batch_predict(texts, endpoint):
    # 构造请求数据
    payload = {"texts": texts}
    headers = {"Content-Type": "application/json"}

    # 发送POST请求到FastAPI服务
    response = requests.post(f"{API_URL}/{endpoint}", data=json.dumps(payload), headers=headers)
    
    # 解析并返回结果
    if response.status_code == 200:
        results = response.json()
        return results
    else:
        raise Exception(f"Request failed with status code {response.status_code}")

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines

def write_results_file(file_path, results):
    with open(file_path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    # 读取文本文件中的文本行
    input_file = "input.txt"
    output_file = "output.txt"
    texts = read_text_file(input_file)

    # 批量调用FastAPI服务并获取结果
    results = batch_predict(texts, "predict_all")
    print(results)

    # 将结果写入输出文件
    write_results_file(output_file, results)
