import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import copy
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score
import matplotlib.pyplot as plt
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv
from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model
import itertools
import logging
from sklearn.metrics import confusion_matrix
from pylab import mpl
from sklearn import manifold
from matplotlib import offsetbox

# 创建一个全局的日志记录器字典
loggers = {}


def configure_logger(log_file, log_name):
    # 如果日志记录器已存在，则返回已存在的日志记录器
    if log_name in loggers:
        return loggers[log_name]

    # 创建一个新的日志记录器
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # 配置日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 配置日志文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 添加文件处理器到日志记录器
    logger.addHandler(file_handler)

    # 将日志记录器添加到全局字典中
    loggers[log_name] = logger

    return logger


def cm_plot(original_label, predict_scores, class_names, save_dir):
    config = {
        "font.family": 'serif',
        "font.size": 6,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],  # 'Microsoft YaHei' SimSun

    }
    mpl.rcParams.update(config)
    mpl.rcParams['font.serif'] = ['SimSun']  # 用来正常显示中文
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 将原始标签列表转换为一个整数张量
    original_label = torch.cat(original_label).cpu().numpy()

    # 将预测分数列表转换为一个张量
    predict_scores_tensor = torch.cat(predict_scores)

    # 使用 torch.argmax 获取预测的类别标签
    predict_label = torch.argmax(predict_scores_tensor, dim=1).cpu().numpy()

    # print('len_predict_label = ', len(predict_label))
    # print('len_label = ', len(original_label))
    #
    # print('predict_label = ', predict_label)
    # print('label = ', original_label)

    cm = confusion_matrix(original_label, predict_label)
    plt.figure(figsize=(14, 14))
    plt.matshow(cm, cmap=plt.cm.Blues)  # 画混淆矩阵，配色风格为cm.Blues
    plt.colorbar()  # 标签颜色
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('真实标签', fontproperties='SimSun')
    plt.xlabel('预测标签', fontproperties='SimSun')

    plt.xticks(fontproperties='SimSun', size=10)
    plt.yticks(fontproperties='SimSun', size=10)
    matrix_output = os.path.join(save_dir, 'Confusion matrix.jpg')
    plt.savefig(matrix_output, dpi=300, bbox_inches='tight')

    # 计算和输出每个类别的准确率
    class_accuracy_info = []
    for i, class_name in enumerate(class_names):
        correct_predictions = cm[i, i]  # 该类别被正确预测的样本数
        total_samples = np.sum(cm[i, :])  # 该类别的总样本数
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        accuracy_str = f'类别 "{class_name}" 的准确率: {accuracy:.2%}'
        print(accuracy_str)
        class_accuracy_info.append(accuracy_str)
        class_accuracy_str = ", ".join(class_accuracy_info)

    matrix_output = os.path.join(save_dir, 'Confusion matrix.jpg')
    plt.savefig(matrix_output, dpi=300, bbox_inches='tight')
    return class_accuracy_str


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs, save_dir):
    f = open(metrics_output, 'a', newline='')
    writer = csv.writer(f)

    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Classes', 'Precision', 'Recall', 'F1 Score', 'Average Precision']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        data.append('{:.2f}'.format(APs[indexs[i]] * 100))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1, TITLE)
    # table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    writer.writerows(TABLE_DATA_1)
    writer.writerow([])
    print()

    TITLE = 'Total Results'
    TABLE_DATA_2 = (
        ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
        ('{:.2f}'.format(eval_results.get('accuracy_top-1', 0.0)),
         '{:.2f}'.format(eval_results.get('accuracy_top-5', 100.0)),
         '{:.2f}'.format(mean(eval_results.get('precision', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('recall', 0.0))),
         '{:.2f}'.format(mean(eval_results.get('f1_score', 0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2, TITLE)
    # table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])
    print()

    writer_list = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    for i in range(len(eval_results.get('confusion'))):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3, TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_3)
    print()


def get_prediction_output(preds, targets, image_paths, classes_names, indexs, prediction_output):
    nums = len(preds)
    f = open(prediction_output, 'a', newline='')
    writer = csv.writer(f)

    results = [['File', 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names)

    for i in range(nums):
        temp = [image_paths[i]]
        pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
        true_label = classes_names[indexs[targets[i].item()]]
        success = True if pred_label == true_label else False
        class_score = preds[i].tolist()
        temp.extend([pred_label, true_label, success])
        temp.extend(class_score)
        results.append(temp)

    writer.writerows(results)


def plot_ROC_curve(preds, targets, classes_names, savedir):
    rows = len(targets)
    cols = len(preds[0])
    ROC_output = os.path.join(savedir, 'ROC')
    PR_output = os.path.join(savedir, 'P-R')
    os.makedirs(ROC_output, exist_ok=True)
    os.makedirs(PR_output, exist_ok=True)
    APs = []
    all_FPR = []
    all_TPR = []
    all_precision = []
    all_recall = []

    for j in range(cols):
        gt, pre, pre_score = [], [], []
        for i in range(rows):
            if targets[i].item() == j:
                gt.append(1)
            else:
                gt.append(0)

            if torch.argmax(preds[i]).item() == j:
                pre.append(1)
            else:
                pre.append(0)

            pre_score.append(preds[i][j].item())

        # ROC
        ROC_csv_path = os.path.join(ROC_output, classes_names[j] + '.csv')
        ROC_img_path = os.path.join(ROC_output, classes_names[j] + '.png')
        ROC_f = open(ROC_csv_path, 'a', newline='')
        ROC_writer = csv.writer(ROC_f)
        ROC_results = []

        FPR, TPR, threshold = roc_curve(targets.tolist(), pre_score, pos_label=j)

        AUC = auc(FPR, TPR)

        ROC_results.append(['AUC', AUC])
        ROC_results.append(['FPR'] + FPR.tolist())
        ROC_results.append(['TPR'] + TPR.tolist())
        ROC_results.append(['Threshold'] + threshold.tolist())
        ROC_writer.writerows(ROC_results)

        all_FPR.append(FPR)
        all_TPR.append(TPR)

        # AP (gt为{0,1})
        AP = average_precision_score(gt, pre_score)
        APs.append(AP)

        # P-R
        PR_csv_path = os.path.join(PR_output, classes_names[j] + '.csv')
        PR_img_path = os.path.join(PR_output, classes_names[j] + '.png')
        PR_f = open(PR_csv_path, 'a', newline='')
        PR_writer = csv.writer(PR_f)
        PR_results = []

        PRECISION, RECALL, thresholds = precision_recall_curve(targets.tolist(), pre_score, pos_label=j)

        PR_results.append(['RECALL'] + RECALL.tolist())
        PR_results.append(['PRECISION'] + PRECISION.tolist())
        PR_results.append(['Threshold'] + thresholds.tolist())
        PR_writer.writerows(PR_results)

        all_precision.append(PRECISION)
        all_recall.append(RECALL)

        plt.figure()
        plt.title(classes_names[j] + ' ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(FPR, TPR, color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(ROC_img_path)

        plt.figure()
        plt.title(classes_names[j] + ' P-R CURVE (AP={:.2f})'.format(AP))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.plot(RECALL, PRECISION, color='g')
        plt.savefig(PR_img_path)

    # 绘制平均 ROC 曲线图
    plt.figure()
    plt.title('Average ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    for fpr, tpr in zip(all_FPR, all_TPR):
        plt.plot(fpr, tpr, alpha=0.5)
    plt.plot([0, 1], [0, 1], color='m', linestyle='--')

    # 保存平均 ROC 曲线图
    plt.savefig(os.path.join(savedir, 'average_ROC_curve.png'))

    # 绘制平均 P-R 曲线图
    plt.figure()
    plt.title('Average P-R Curve (mAP={:.2f})'.format(np.mean(APs)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    for precision, recall in zip(all_precision, all_recall):
        plt.plot(recall, precision, alpha=0.5)

    # 保存平均 P-R 曲线图
    plt.savefig(os.path.join(savedir, 'average_PR_curve.png'))

    return APs


def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(4, 3))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 6.), #colormap 返回颜色
                 fontdict={'weight': 'bold', 'size': 6})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
    plt.axis([0,1.1,0,1.1])
    plt.xticks([]), plt.yticks([])#传递空列表来禁用x，y刻度
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
    if title is not None:
        plt.title(title,fontsize=10)

def tsne(predict_test_array, savedir):
    label = [np.argmax(item) for item in predict_test_array]
    X = predict_test_array
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    print(X_tsne.shape)

    plot_embedding(X_tsne,label,
                   "Output of model")
    plt.savefig(os.path.join(savedir, '模型输出聚类图.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


# def main():
#     args = parse_args()
#     model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
#
#     """
#     创建评估文件夹、metrics文件、混淆矩阵文件
#     """
#     dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#     save_dir = os.path.join('eval_results', model_cfg.get('backbone').get('type'), dirname)
#     metrics_output = os.path.join(save_dir, 'metrics_output.csv')
#     prediction_output = os.path.join(save_dir, 'prediction_results.csv')
#     os.makedirs(save_dir)
#
#     """
#     获取类别名以及对应索引、获取标注文件
#     """
#     classes_map = 'datas/annotations.txt'
#     test_annotations = 'datas/test.txt'
#     classes_names, indexs = get_info(classes_map)
#     with open(test_annotations, encoding='utf-8') as f:
#         test_datas = f.readlines()
#
#     """
#     生成模型、加载权重
#     """
#     if args.device is not None:
#         device = torch.device(args.device)
#     else:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = BuildNet(model_cfg)
#     if device != torch.device('cpu'):
#         model = DataParallel(model, device_ids=[args.gpu_id])
#     model = init_model(model, data_cfg, device=device, mode='eval')
#
#     """
#     制作测试集并喂入Dataloader
#     """
#     val_pipeline = copy.deepcopy(train_pipeline)
#     test_dataset = Mydataset(test_datas, val_pipeline)
#     test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
#                              num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)
#
#     # 在训练中使用train_logger记录训练信息
#     test_log_file = os.path.join(save_dir, 'test.txt')
#     test_logger = configure_logger(test_log_file, 'test')
#
#     """
#     计算Precision、Recall、F1 Score、Confusion matrix
#     """
#     with torch.no_grad():
#         preds, targets, image_paths = [], [], []
#         with tqdm(total=len(test_datas) // data_cfg.get('batch_size')) as pbar:
#             for _, batch in enumerate(test_loader):
#                 images, target, image_path = batch
#                 outputs = model(images.to(device), return_loss=False)
#                 preds.append(outputs)
#                 targets.append(target.to(device))
#                 image_paths.extend(image_path)
#                 pbar.update(1)
#
#     eval_results = evaluate(torch.cat(preds), torch.cat(targets), data_cfg.get('test').get('metrics'),
#                             data_cfg.get('test').get('metric_options'))
#     preds_list =  [tensor.cpu().numpy() for tensor in preds]
#     preds_array = np.array(preds_list)
#     print('preds_array.shape = ', preds_array.shape)
#     print('preds_array = ', preds_array)
#     tsne(preds_array, save_dir)
#
#     APs = plot_ROC_curve(torch.cat(preds), torch.cat(targets), classes_names, save_dir)
#     get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs, save_dir)
#     get_prediction_output(torch.cat(preds), torch.cat(targets), image_paths, classes_names, indexs, prediction_output)
#     class_accuracy_str = cm_plot(targets, preds, classes_names, save_dir)
#
#     # 打印验证结果到验证日志文件
#     precision_percentage = eval_results.get('accuracy_top-1', 0.0) / 100.0
#     test_logger.info(f"Precision = {precision_percentage:.2%}, {class_accuracy_str}")
#     logging.shutdown()

def main():
    args = parse_args()
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('eval_results', model_cfg.get('backbone').get('type'), dirname)
    metrics_output = os.path.join(save_dir, 'metrics_output.csv')
    prediction_output = os.path.join(save_dir, 'prediction_results.csv')
    os.makedirs(save_dir)

    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_map = 'datas/annotations.txt'
    test_annotations = 'datas/test.txt'
    classes_names, indexs = get_info(classes_map)
    with open(test_annotations, encoding='utf-8') as f:
        test_datas = f.readlines()

    """
    生成模型、加载权重
    """
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BuildNet(model_cfg)
    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])
    model = init_model(model, data_cfg, device=device, mode='eval')

    """
    制作测试集并喂入Dataloader
    """
    val_pipeline = copy.deepcopy(train_pipeline)
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'),
                             num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)

    # 在训练中使用train_logger记录训练信息
    test_log_file = os.path.join(save_dir, 'test.txt')
    test_logger = configure_logger(test_log_file, 'test')

    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    with torch.no_grad():
        preds, targets, image_paths = [], [], []
        with tqdm(total=len(test_datas) // data_cfg.get('batch_size')) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                outputs = model(images.to(device), return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.extend(image_path)
                pbar.update(1)

    eval_results = evaluate(torch.cat(preds), torch.cat(targets), data_cfg.get('test').get('metrics'),
                            data_cfg.get('test').get('metric_options'))
    preds_tensor = torch.cat(preds)
    preds_array = preds_tensor.cpu().numpy()
    print('preds_array.shape = ', preds_array.shape)
    print('preds_array = ', preds_array)
    tsne(preds_array, save_dir)

    APs = plot_ROC_curve(torch.cat(preds), torch.cat(targets), classes_names, save_dir)
    get_metrics_output(eval_results, metrics_output, classes_names, indexs, APs, save_dir)
    get_prediction_output(torch.cat(preds), torch.cat(targets), image_paths, classes_names, indexs, prediction_output)
    class_accuracy_str = cm_plot(targets, preds, classes_names, save_dir)

    # # 打印验证结果到验证日志文件
    # precision_percentage = eval_results.get('accuracy_top-1', 0.0) / 100.0
    # test_logger.info(f"Precision = {precision_percentage:.2%}, {class_accuracy_str}")
    # logging.shutdown()

    # 计算所有类别的平均准确率
    accuracies = []
    for i, class_name in enumerate(classes_names):
        correct_predictions = eval_results.get('confusion')[i][i]  # 获取混淆矩阵中的正确预测数
        total_samples = np.sum(eval_results.get('confusion')[i, :])  # 获取该类别的总样本数
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        accuracies.append(accuracy)

    # 计算平均准确率
    average_precision = np.mean(accuracies)
    # 打印验证结果到验证日志文件
    test_logger.info(f"Precision = {average_precision:.2%}, {class_accuracy_str}")
    logging.shutdown()

if __name__ == "__main__":
    main()
