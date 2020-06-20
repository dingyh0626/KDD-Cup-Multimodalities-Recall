## KDD-Cup-Multimodalities-Recall

[KDD-Cup-Multimodalities-Recall](https://tianchi.aliyun.com/competition/entrance/231786/rankingList/1)
第10名来自垫底小分队的方案。ndcg@5指标：A榜单模model1得分0.6969，双模集成得分0.7158；B榜双模集成得分0.7276。

git链接：[https://github.com/IntoxicatedDING/KDD-Cup-Multimodalities-Recall.git](https://github.com/IntoxicatedDING/KDD-Cup-Multimodalities-Recall.git)，阿里云code链接（包含部分必要预处理数据）：[https://code.aliyun.com/zjhndyhnba/KDD-Cup-Multimodalities-Recall-FINAL.git](https://code.aliyun.com/zjhndyhnba/KDD-Cup-Multimodalities-Recall-FINAL.git)


### 方案
#### 预训练（[`multilabel`](multilabel)）：
- 对每个query文本进行分词，并去除停用词，每个词作为一个标签。
- 以图片作为输入，进行多标签分类。
- 以图片作为输入，进行查询文本生成（image caption）。
- 其中多标签分类和查询文本生成共同进行训练。

#### 基于单词（标签）与图片的匹配模型（[`model1`](model1)）：
- 使用预训练的图片embedding（多标签分类任务得到）和查询文本中的每个单词计算匹配分，对图片的encoder进行fine tuning。
- 对每个单词生成一个权重。
- 使用上述权重对单词匹配分加权平均得到最终得分。
- 训练方式为pairwise。

#### 基于句子与图片的匹配模型（[`model2`](model2)）
- 使用预训练的图片embedding（查询文本生成任务得到）和整个查询文本计算匹配分，对图片的encoder进行fine tuning。
- 训练方式为pointwise。

#### 集成（[`ensemble.ipynb`](ensemble.ipynb)）
- 对上述两个模型的得分进行加权平均作为最终得分。
- 权重通过在验证集上进行搜索得到。

### 目录结构
```
|-- data
	|-- multimodal_labels.txt
	|-- train
		|-- train.tsv
	|-- valid
		|-- valid.tsv
		|-- valid_answer.json
	|-- testA
		|-- testA.tsv
	|-- testB
		|-- testB.tsv
	|-- info
	    |-- data.pkl（剥除base64数据）
	    |-- data_info2.pkl（聚类、字典等信息）
	    |-- query2product2.pkl（查询文本到物品的映射）
	    
|-- user_data
    |-- image_encoder_large.pth（图片编码器预训练模型，未上传）

|-- external_resources
    |--test_pred_model1.json
    |--valid_pred_model1.json
    |--test_pred_model2.json
    |--valid_pred_model2.json
    |--submission.csv
	 
```


### 流程
1. 数据准备工作：
```
tar -zxvf info.tar.gz -C ./data
python preprocess.py
```

2. 进入[`multilabel`](multilabel)目录使用`train.py`脚本进行训练，完成后使用`export_model.py`将`ImageEncoder`预训练模型导出：
```
cd multilabel
python train.py
python export_model.py --epoch 6
```

3. 进入[`model1`](model1)目录使用`train.py`脚本进行训练，完成后使用`validation.py`输出测试数据集的预测：
```
cd model1
python -u -m torch.distributed.launch --nproc_per_node=2 train.py --devices 0 1
python validation.py --epoch 5
```

4. 进入[`model2`](model2)目录使用`train.py`脚本进行训练，完成后使用`validation.py`输出测试数据集的预测：
```
cd model2
python -u -m torch.distributed.launch --nproc_per_node=2 train.py --devices 0 1
python validation.py --epoch 2
```

5. 集成：
```
python ensemble.py
```




### 测试
修改相关文件的路径，执行如下命令：
```
tar -zxvf info.tar.gz -C ./data
cd model1
tar -zxvf ckpt.tar.gz
python validation.py
cd ../model2
tar -zxvf ckpt.tar.gz
python validation.py
cd ..
python ensemble.py
```


### 环境
- torch==1.3.1
- prefetch-generator==1.0.1
- transformers==2.8.0
- numpy==1.17.2
