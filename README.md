# Distributionally Robust Optimization for Unbiased Learning to Rank

This repository contains the supporting code for our paper "Distributionally Robust Optimization for Unbiased Learning to Rank".
The implementation for training and evaluating ULTR models is based on the [ULTRA_pytorch](https://github.com/ULTR-Community/ULTRA_pytorch) toolkit.

## Repository Usage
### Get Started
```
pip install -r requirements.txt
```

### Preparing dataset
In our paper, we utilized a public datasets: [Baidu-ultr-uva](https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr).
For reproducibility, one can download the dataset and place it at [DATA_PATH].

### K-Means Clustering
To obtain K-Means cluster centors for training query-document pairs:
```
python count_baidu_tra_cluster.py [DATA_PATH] [CLUSTER_NUM]
```

To obtain K-Means cluster centors for training sessions:
```
python count_baidu_tra_qpp_cluster.py [DATA_PATH] [CLUSTER_NUM]
```

### Training and evaluating DRO-ULTR models
To train the DRO-PBM method using clicks:
```
python main_tra_cluster.py --ULTR_model DRO-PBM --dataset_dir [DATA_PATH] --model_dir [SAVE_MODEL_PATH] --setting_file ./example_settings/naive_exp_settings.json
```
To train the DRO-Regression-EM method using clicks:
```
python main_tra_cluster.py --ULTR_model DRO-Regression-EM --dataset_dir [DATA_PATH] --model_dir [SAVE_MODEL_PATH] --setting_file ./example_settings/naive_exp_settings.json
```
To train the DRO-IPS method using clicks:
```
python main_tra_qpp_cluster.py --ULTR_model DRO-IPS --dataset_dir [DATA_PATH] --model_dir [SAVE_MODEL_PATH] --setting_file ./example_settings/naive_exp_settings.json
```
To train the DRO-DLA method using clicks:
```
python main_tra_qpp_cluster.py --ULTR_model DRO-DLA --dataset_dir [DATA_PATH] --model_dir [SAVE_MODEL_PATH] --setting_file ./example_settings/naive_exp_settings.json
```
To evaluate a ULTR method:
```
python main_tra_cluster.py --dataset_dir [DATA_PATH]  --setting_file ./example_settings/test_exp_settings.json --batch_size 1 --test_only True  --model_dir [MODEL_PATH]
```