# -*- coding: utf-8 -*-
"""
画图函数
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'Agg'
import seaborn as sns
from sklearn.metrics import roc_curve,confusion_matrix,auc
import itertools
from .tools import best_prob

class PlotFeatureEn(object):
    """特征工程作图函数"""

    def draw_IV(self,IV_dict, path, xlabel=None, figsize=(15, 7), is_save=False):
        """
        信息值IV柱状图
        ---------------------
        param
        IV_dict: dict IV值字典
        path: str 文件存储地址
        xlabel: list x轴标签
        figsize: tupe 图片大小
        _____________________
        return
        draw_iv
        """
        IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)
        ivlist = [i[1] for i in IV_dict_sorted]
        index = [i[0] for i in IV_dict_sorted]
        fig1 = plt.figure(figsize=figsize)
        ax1 = fig1.add_subplot(1, 1, 1)
        x = np.arange(len(index)) + 1
        ax1.bar(x, ivlist, width=0.5)  # 生成柱状图
        ax1.set_xticks(x)
        if xlabel:
            ax1.set_xticklabels(index, rotation=0, fontsize=8)

        ax1.set_ylabel('IV(Information Value)', fontsize=14)
        # 在柱状图上添加数字标签
        for a, b in zip(x, ivlist):
            plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)

        if is_save:
            plt.savefig(path + "high_iv.png")
        plt.show()
        plt.close()

    def draw_importance(self,importance, features, figsize, path):
        """特征重要度"""
        plt.style.use('fivethirtyeight')
        plt.rcParams['figure.figsize'] = figsize
        sns.set_style("darkgrid", {"font.sans-serif": ["simhei", "Arial"]})
        indices = np.argsort(importance)[::-1]
        plt.figure(figsize=figsize)
        plt.title("feature importance")
        plt.bar(range(len(indices)), importance[indices], color='lightblue', align="center")
        plt.step(range(len(indices)), np.cumsum(importance[indices]), where='mid', label='Cumulative')
        plt.xticks(range(len(indices)), features[indices], rotation='vertical', fontsize=14)
        plt.xlim([-1, len(indices)])
        plt.savefig(path + "feature_importance.png")
        plt.show()
        plt.close()

    def draw_corr(self,df, figsize: tuple, path: str):
        """
        特征相关系数
        ------------------------
        parameter:
        data_new: dataFrame,columns must be number
        figsize: tupe,two number
        return:
                heatmap
        """
        # 相关系数分析
        colormap = plt.cm.viridis
        plt.figure(figsize=figsize)
        plt.title('皮尔森相关性系数', y=1.05, size=8)
        mask = np.zeros_like(df.corr(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True,
                    mask=mask)
        plt.savefig(path + "feature_corr.png")
        plt.show()
        plt.close()

class PlotModel(object):
    """模型作图"""
    def __init__(self,y_true,y_prob):
        """
        :param y_true:array, 真实y
        :param y_prob: array, 预测概率y
        """
        self.y_true = y_true
        self.y_prob = y_prob

    def plot_roc_curve(self,filename='./'):
        """
        """
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        c_stats = auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.plot(fpr, tpr, label="ROC curve")
        auc_value = "AUC = %.3f" % c_stats
        plt.text(0.8, 0.2, auc_value, bbox=dict(facecolor='r', alpha=0.5))
        plt.xlabel('False positive rate')  # 假正率
        plt.ylabel('True positive rate')  # 真正率
        plt.title('ROC curve')  # ROC 曲线
        plt.legend(loc='best')
        plt.savefig(filename+"roc_curve.png")
        plt.show()
        plt.close()
        return auc_value

    def plot_ks_curve(self,filename='./'):
        """
        """
        fpr, tpr, thr = roc_curve(self.y_true, self.y_prob)  # 假正率 真正率 概率阈值
        thr = np.array(sorted(thr))
        ks = abs(fpr - tpr)  # ks 序列
        ks_value = "KS = %.3f" % max(ks)  # ks值
        cut_prob = best_prob(self.y_true, self.y_prob)  # 最佳切分概率
        plt.plot(thr, fpr, label='cum_good', color='blue', linestyle='-', linewidth=2)  # 假正率 累计好
        plt.plot(thr, tpr, label='cum_bad', color='red', linestyle='-', linewidth=2)  # 真正率,累计坏
        plt.plot(thr, ks, label='ks', color='green', linestyle='-', linewidth=2)  # ks曲线
        plt.plot(thr, [max(ks)] * len(thr), color='green', linestyle='--')  # ks值直线
        plt.axvline(cut_prob, color='gray', linestyle='--')  # 最佳切分概率直线
        plt.title('{}'.format(ks_value), fontsize=15)
        plt.xlim((0.0, 1))
        plt.savefig(filename+"ks_curve.png")  # 保存
        plt.show()
        plt.close()
        return ks_value

    def plot_confusion_matrix(self,labels:list,normalize=False,filename='./'):
        """
        混淆矩阵
        ------------------------------------------
        Params
        labels: list, labels class
        normalize: bool, True means trans results to percent
        """
        cut_prob = best_prob(self.y_true, self.y_prob)  # 最佳切分概率
        y_pred = np.array([1 if i >= cut_prob else 0 for i in self.y_prob])
        matrix = confusion_matrix(y_true=self.y_true, y_pred=y_pred, labels=labels)
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues_r)  # 在指定的轴上展示图像
        plt.colorbar()  # 增加色柱
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)  # 设置坐标轴标签
        plt.yticks(tick_marks, labels)

        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, matrix[i, j], fontsize=12,
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title("confusion matrix")
        plt.savefig(filename+"matrix.png")
        plt.show()
        plt.close()