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

#### 流程
1. 进入[`multilabel`](multilabel)目录使用`train.py`脚本进行训练，完成后使用`export_model.py`将`ImageEncoder`预训练模型导出。
2. 进入[`model1`](model1)目录使用`train.py`脚本进行训练，完成后使用`validation.py`输出测试数据集的预测。
3. 进入[`model2`](model2)目录使用`train.py`脚本进行训练，完成后使用`validation.py`输出测试数据集的预测。
4. 使用jupyter运行完[`ensemble.ipynb`](ensemble.ipynb)的代码，输出最后的召回结果。

**注意：** 所有输入目录、输出目录以及`ImageEncoder`加载的预训练模型路径均采用了硬编码，请自行修改。

### 其他说明
- [`info`](info)目录存放一些预处理数据，包括查询文本到图片集合的映射、文本的聚类
（根据文本最后一个单词进行归类以及基于早期模型训练得到的文本embedding的K-Means聚类，目的是用于困难样本挖掘）等信息。
- 并行训练启动方式：`python -u -m torch.distributed.launch --nproc_per_node=${number of GPUs} train.py`
- `utils.py`文件中对训练数据的读取代码需要自行修改。为了避免一次性读入120G数据，我们将原本训练集中的base64字符串全部存成单个文件，
即一个product的base64存成一个文件，文件名为product的id，而读入的训练数据
`info/data.pkl` 只包含文本以及product的索引信息。只有在product轮到训练时，从硬盘读入内存。
实验发现，只要product存到固态硬盘，可在节约内存同时保证GPU利用率。
- product数据使用`np.savez_compressed`函数进行存储：`np.savez_compressed('/.../{product_id}.npz', features=features, boxes=boxes)`，
其中`features`为base64解码的形如[num_obj, 2048]的`ndarray`，同样`boxes`为形如[num_obj, 4]的`ndarray`，读取代码如下：

```python
data = np.load('/.../{product_id}.npz')
features = data['features']
boxes = data['boxes']
```
- 为方便跑通，使用`utils_backup.py`将训练集直接加载到内存（未经测试，可能存在bug）。

