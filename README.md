 Food-ICP(测试方法及测试结果)
====
  一、测试方法
----
### 1、安装环境与依赖
进入项目目录并安装依赖：
- `cd /root/Food-ICP`
- `pip install -r requirements.txt`
### 2、准备数据集
准备数据集时应在根目录创建`datasets`文件夹，并按以下结构准备数据集
```text
|-train
|   |-奥尔良烤鸡中翅
|   |   134959.jpg
|   |   ...
|   |-鸡
|   |   092028.jpg
|   |   ...
|-test
|   |-奥尔良烤鸡中翅
|   |   XXXXXX.jpg
|   |   ...
|-validation
|   |-奥尔良烤鸡中翅
|   |   xxxxxx.jpg
|   |   ...
```
在`Food-ICP/datas/`中创建标签文件`annotations.txt`，按行将`类别名 索引`写入文件；
```text
奥尔良烤鸡中翅 0
奥尔良排骨 1
奥尔良散排 2
白水饺 3
彩水饺 4
...
```
### 3、测试方法
- 确认`Food-ICP/datas/annotations.txt`标签准备完毕
- 确认`Food-ICP/datas/`下`test.txt`与`annotations.txt`对应
- 在`Food-ICP/models/resnet`下找到对应配置文件`resnet50.py`
- 修改配置文件参数，主要修改权重路径
- 在`Food-ICP`打开终端运行
`python tools/evalution.py models/resnet/resnet50.py`
### 4、测试输出目录
测试结果将输出在`Food-ICP/eval_results/`下，若没有此文件夹请创建此文件夹

 二、测试结果
 ----
输出的测试结果将包含以下几个部分
```text
eval_results/
└── <backbone_type>/
    └── <YYYY-MM-DD-HH-MM-SS>/
        ├── test.txt                      # 测试日志（含各类别准确率等）
        ├── metrics_output.csv            # 汇总指标：Precision/Recall/F1/AP + Top1/Top5 + 混淆矩阵表
        ├── prediction_results.csv        # 每张图片的预测明细：预测/真值/是否正确 + 各类别得分
        ├── Confusion matrix.jpg          # 混淆矩阵图
        ├── 模型输出聚类图.png               # t-SNE 聚类可视化
        ├── average_ROC_curve.png         # 平均 ROC 曲线
        ├── average_PR_curve.png          # 平均 PR 曲线
        ├── ROC/                          # 每个类别的 ROC 曲线与数据
        │   ├── <class_name>.csv
        │   └── <class_name>.png
        └── P-R/                          # 每个类别的 PR 曲线与数据
            ├── <class_name>.csv
            └── <class_name>.png
```
部分测试结果如下：
### 1、模型输出聚类图
<img width="400" height="300" alt="模型输出聚类图" src="https://github.com/user-attachments/assets/82ffdf55-19b6-4e28-a597-d6e62f376b28" />

### 2、average_PR_curve
<img width="640" height="480" alt="average_PR_curve" src="https://github.com/user-attachments/assets/c8f25717-a063-4ce8-adcc-d29034770f54" />

### 3、average_ROC_curve
<img width="640" height="480" alt="average_ROC_curve" src="https://github.com/user-attachments/assets/86c35655-7add-4f6e-99c1-6ac1b2471169" />

### 4、Confusion matrix
![Confusion matrix](https://github.com/user-attachments/assets/528ce3b7-264c-4502-bbc3-218c4c6f3862)
