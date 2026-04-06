import json
import os
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='要评估的JSON文件路径')
    parser.add_argument('--backbone',type=str)
    return parser.parse_args()



def parse_filename(filename):
    parts = filename.replace('.json', '').split('_')
    method = parts[0]
    dataset = parts[1]
    lang = parts[2]
    result_num = parts[-1]
    return method, dataset, lang, result_num


def organize_results(results_dict):
    methods = sorted(set(k[1] for k in results_dict.keys()))
    metrics = ['Reliability', 'Generality', 'Locality', 'Portability']

    index = pd.MultiIndex.from_product([methods, metrics], names=['Method', 'Metric'])
    columns = []

    for lang in sorted(set(k[0] for k in results_dict.keys())):
        columns.extend([f'{lang}_F1', f'{lang}_EM'])

    df = pd.DataFrame(index=index, columns=columns)

    for (lang, method, metric) in results_dict:
        f1_score, em_score = results_dict[(lang, method, metric)].split('/')
        df.loc[(method, metric), f'{lang}_F1'] = float(f1_score)
        df.loc[(method, metric), f'{lang}_EM'] = float(em_score)

    return df


def obtain_f1_and_em(a, b):
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/models/xlm-roberta-base")

    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em


def my_avg(a):
    return round(sum(a) * 100 / float(len(a)), 2)


def calculate_metrics(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reliability_f1, reliability_em = [], []
    generalization_f1, generalization_em = [], []
    locality_f1, locality_em = [], []
    portability_f1, portability_em = [], []

    print(f"Processing {len(data)} items from {file_path}...")

    for item in tqdm(data, desc="Processing data", unit="item"):
        f1, em = obtain_f1_and_em(item["post"]["reliability"]["ans"], item["post"]["reliability"]["target"])
        reliability_f1.append(f1)
        reliability_em.append(em)

        f1, em = obtain_f1_and_em(item["post"]["generalization"]["ans"], item["post"]["generalization"]["target"])
        generalization_f1.append(f1)
        generalization_em.append(em)

        f1, em = obtain_f1_and_em(item["post"]["locality"]["neighborhood_acc"]["ans"],
                                  item["pre"]["locality"]["neighborhood_acc"]["ans"])
        locality_f1.append(f1)
        locality_em.append(em)

        f1, em = obtain_f1_and_em(item["post"]["portability"]["one_hop_acc"]["ans"],
                                  item["post"]["portability"]["one_hop_acc"]["target"])
        portability_f1.append(f1)
        portability_em.append(em)

    print("Processing complete!")

    results = {
        "reliability": f"{my_avg(reliability_f1)}/{my_avg(reliability_em)}",
        "generalization": f"{my_avg(generalization_f1)}/{my_avg(generalization_em)}",
        "locality": f"{my_avg(locality_f1)}/{my_avg(locality_em)}",
        "portability": f"{my_avg(portability_f1)}/{my_avg(portability_em)}"
    }

    print("Processing avg complete!")
    return results


def main():
    args = parse_args()
    file_path = args.file

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    # 解析文件名
    filename = os.path.basename(file_path)
    method, dataset, lang, _ = parse_filename(filename)

    # 计算指标
    metrics = calculate_metrics(file_path)

    results_dict = {}
    for metric_name, score in metrics.items():
        metric_map = {
            'reliability': 'Reliability',
            'generalization': 'Generality',
            'locality': 'Locality',
            'portability': 'Portability'
        }
        results_dict[(lang, method, metric_map[metric_name])] = score

    # 生成表格
    df = organize_results(results_dict)

    os.makedirs('./csv-results/', exist_ok=True)
    output_file = f'./csv-results/{args.backbone}_{method}_{dataset.lower()}_result.csv'
    df.to_csv(output_file)
    print(f"结果已保存到：{output_file}")

    print("\n📊 评估结果：")
    print(df)


if __name__ == "__main__":
    main()