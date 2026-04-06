import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument("--lang1", type=str, default="")
    parser.add_argument("--lang2", type=str, default="")
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams  # editing_hparams 是一个类的对象
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    else:
        raise NotImplementedError
        
    # BMIKE-53/baselines/data/BMIKE53/zsRE/zsre_test_zh.json

    test_data = json.load(open(f'./data/zsre/zsre_test_zh.json', 'r', encoding='utf-8'))

    # 随机采样
    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)
        ds_size = args.ds_size
    else:
        ds_size = 0

    #filter out data which don't have lang2
    test_data = [data for data in test_data if args.lang2 in data]

    prompts_truth = [test_data_[args.lang1]['src'] for test_data_ in test_data] # lang1的所有问题列表
    # print("1111111")
    # for test_data_ in test_data:
    #     print(test_data_)
    #     print(test_data_[args.lang2])
    #     print("test_data_[args.lang2]['src']" + test_data_[args.lang2]['src'])
    #     print("22222")
    # print("333333")
    prompts_test = [test_data_[args.lang2]['src'] for test_data_ in test_data] # lang2的所有问题列表
    target_truth = [edit_data_[args.lang1]['alt'] for edit_data_ in test_data] # lang1的所有答案列表
    target_test = [edit_data_[args.lang2]['alt'] for edit_data_ in test_data] # test in chinese lang2的所有答案列表
    rephrase_prompts = [edit_data_[args.lang2]['rephrase'] for edit_data_ in test_data]
    locality_prompts = [edit_data_[args.lang2]['loc'] for edit_data_ in test_data]
    locality_ans = [edit_data_[args.lang2]['loc_ans'] for edit_data_ in test_data]
    portability_prompts = [edit_data_[args.lang2]['port'] for edit_data_ in test_data]
    portability_ans = [edit_data_[args.lang2]['port_ans'] for edit_data_ in test_data]
    subject = [edit_data_[args.lang1]['subject'] for edit_data_ in test_data]

    edited_inputs = {
        'edited_english': {
            'prompt': prompts_truth,
            'ground_truth': target_truth
        },
    }
    cross_inputs = {
        'cross': {
            'prompt': prompts_test,
            'ground_truth': target_test
        },
    }
    generalization_inputs = {
        'rephrase': {
            'prompt': rephrase_prompts,
            'ground_truth': target_test
        },
    }
    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }
    hparams = editing_hparams.from_hparams(args.hparams_dir) # 以rome为例，hparams_dir：baselines/hparams/ROME/llama3.2-3b.yaml

    # if args.editing_method == 'IKE':
    #     train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
    #     train_ds = ZsreDataset(train_data_path)
    #     sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    #     encode_ike_facts(sentence_model, train_ds, hparams)
    # else:
    #     train_ds = None

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts_truth,
        edited_inputs=edited_inputs,
        cross_inputs=cross_inputs,
        generalization_inputs=generalization_inputs,
        # rephrase_prompts=rephrase_prompts,
        target_new=target_truth,
        subject=subject,
        # train_ds=train_ds,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )
    # 告诉editor要改什么知识（prompts，target_new）
    # 用哪些测试集验证编辑效果（edited /cross/generalization /locality/portability）

    os.makedirs(args.metrics_save_dir, exist_ok=True)

    # json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
    # Construct the file path
    file_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.data_dir[7:11]}_{args.lang1}_{args.lang2}_results_{ds_size}.json')

    # ./output
    # Save the metrics
    with open(file_path, 'w',encoding='utf-8') as file:
        json.dump(metrics, file, indent=4,ensure_ascii=False)