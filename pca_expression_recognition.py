import os
import numpy as np
import mahotas as mh    ##图像预处理
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 忽略 警告
import warnings
warnings.filterwarnings("ignore")

root = r"D:\dataset\expression"
# 表情→数字映射
label_dict = {"angry":0,"disgust":1,"fear":2,"happy":3,"neutral":4,"sad":5,"surprise":6}
emotion_name = ["angry","disgust","fear","happy","neutral","sad","surprise"]

X_train, y_train = [], []
X_test, y_test = [], []

# 把照片导入Numpy数组，然后把它们的像素矩阵转换成向量：
for dataset in ["train", "test"]:
    dataset_path = os.path.join(root, dataset)
    #遍历7个表情文件夹
    for cls_name in os.listdir(dataset_path):     #cls_name开始遍历各类emotion
        if cls_name not in label_dict:
            continue
        label = label_dict[cls_name]
        cls_path = os.path.join(dataset_path, cls_name)     #拼接
        # 遍历图片
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            arr = scale(mh.imread(img_path, as_grey=True).reshape(48*48).astype('float32'))
            if dataset == "train":
                X_train.append(arr)
                y_train.append(label)
            else:
                X_test.append(arr)
                y_test.append(label)

# 列表转numpy数组
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 用交叉检验建立训练集和测试集，在训练集上用PCA
pca = PCA(n_components=500)  # 30   50    500

# 训练一个逻辑回归分类器。scikit-learn底层会自动用one versus all策略创建二元分类器：
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print('训练集数据的原始维度是：{}'.format(X_train.shape))
print('PCA降维后训练集数据是：{}'.format(X_train_reduced.shape))
classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)

# 最后，用交叉验证和测试集评估分类器的性能。

print('交叉验证准确率是：{}\n{}'.format(np.mean(accuracies), accuracies))
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print(classification_report(y_test,predictions,target_names=emotion_name))