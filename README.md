# multilingualKE

这是一个用于跨语言知识编辑实验的项目，基于 ZSRE数据集，实现了多种知识编辑方法在英语与中文的跨语言场景下的评估和测试。

## 运行示例

1. 使用run_zsre.py进行知识编辑

```
python run_zsre.py --editing_method FT --hparams_dir ./hparams/FT/bloomz-1b7 --data_dir ./data/zsre/zsre_test_ --lang1 en --lang2 zh --ds_size 3 --metrics_save_dir ./output/bloomz-1b7/ft
```
参数说明：

- editing_method: 编辑方法选择
- hparams_dir: 超参数配置文件路径
- data_dir: 数据目录
- lang1: 编辑语言
- lang2: 测试语言
- ds_size: 数据集大小（可选）
- metrics_save_dir结果保存的目录

2. 对编辑结果进行测试
```
python evaluate_test.py --file ./output/baichuan-7b/ft/FT_zsre_en_en_results_3.json  --backbone
```
参数说明：

- --file：要评估的JSON文件路径
- --backbone：文件名标识符

## 数据格式

本项目使用JSON格式的数据文件，结构如下：

    {
      "case_id": {
        "en": {
          "src": "英语问题",
          "alt": "英语答案",
          "subject": "主题",
          "rephrase": "重述问题",
          "loc": "局部性问题",
          "loc_ans": "局部性答案",
          "port": "可移植性问题",
          "port_ans": "可移植性答案"
        },
        "zh": {
          "src": "中文问题",
          "alt": "中文答案",
          ...
        }
      }
    }



## 配置说明

超参数配置

每种编辑方法都有对应的YAML配置文件，位于hparams/目录下。主要配置项包括：

- 模型参数（模型名称、设备）
- 训练参数（学习率、批次大小、轮数）
- 编辑参数（层数、秩等）
- 评估参数


