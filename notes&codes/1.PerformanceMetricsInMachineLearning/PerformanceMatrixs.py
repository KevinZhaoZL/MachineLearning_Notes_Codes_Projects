import numpy as np

y_true = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
######################################################################
## 1.错误率和精度
precision = np.mean(y_pred == y_true)
error = 1 - precision
print(precision, error)
from sklearn.metrics import accuracy_score

# 返回准确率
precision = accuracy_score(y_true, y_pred, normalize=True)
# 返回正确分类的数量
precision_num = accuracy_score(y_true, y_pred, normalize=False)
print(precision, precision_num)
######################################################################
## 2. 查准率、查全率、F-Score
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(precision, recall)
## 3. P-R曲线的绘制
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
y = label_binarize(y, classes=[0, 1, 2])  # one-hot
n_classes = y.shape[1]
# 添加噪声
np.random.seed(0)
n_samples, n_features = X.shape
X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]
# 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
clf.fit(X_train, y_train)
y_score = clf.fit(X_train, y_train).decision_function(X_test)
# 绘制P-R曲线
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
precision = {}
recall = {}
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    ax.plot(recall[i], precision[i], label='target=%s' % i)
ax.set_xlabel("Recall Score")
ax.set_ylabel("Precision Score")
ax.set_title("P-R")
ax.legend(loc='best')
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.grid()
plt.show()
## 4.F-Score
from sklearn.metrics import f1_score, fbeta_score

f1S = f1_score(y_true, y_pred)
fbS = fbeta_score(y_true, y_pred, beta=100)
print(f1S, fbS)
##################################################################
## 5. ROC与AUC
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target
# one-hot
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
# 添加噪声
np.random.seed(0)
n_samples, n_features = X.shape
X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# 训练模型
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0))
clf.fit(X_train, y_train)
y_score = clf.fit(X_train, y_train).decision_function(X_test)
# 获取ROC
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax.plot(fpr[i], tpr[i], label="target=%s,auc=%s" % (i, roc_auc[i]))
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("ROC")
ax.legend(loc="best")
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.grid()
plt.show()
###############################################################
## 6. IoU
import numpy as np


def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上和右下的坐标
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ## 计算交集面积
    inter_area = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1])

    ## 计算交并比
    IoU = inter_area / (area1 + area2 - inter_area)
    return IoU
