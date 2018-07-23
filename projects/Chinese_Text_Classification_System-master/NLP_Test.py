# -*- coding: utf-8 -*-
import os
import jieba
import pickle

from sklearn import metrics
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from numpy import *


class NLP_C:
    def __init__(self):
        self.corpus_path_train = "data/train/"
        self.seg_path_train = "data/train_seg/"
        self.wordbag_path_train = "data/train_word_bag/train_set.dat"
        self.corpus_path_test = "data/test/"
        self.seg_path_test = "data/test_seg/"
        self.wordbag_path_test = "data/test_word_bag/test_set.dat"
        self.stopword_path = "data/train_word_bag/hlt_stop_words.txt"
        self.space_path_train = "data/train_word_bag/tfidfspace.dat"
        self.space_path_test = "data/test_word_bag/testspace.dat"

    def savefile(self, savepath, content):
        fp = open(savepath, 'w', encoding='gb2312', errors='ignore')
        fp.write(content)
        fp.close()

    def readfile(self, path):
        fp = open(path, 'r', encoding='gb2312', errors='ignore')
        content = fp.read()
        fp.close()
        return content

    def splitwords(self, corpus_path, seg_path):
        # 获取该目录下所有子目录
        catelist = os.listdir(corpus_path)

        for mydir in catelist:
            class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径
            seg_dir = seg_path + mydir + "/"  # 拼出分词后的语料分类目录
            if not os.path.exists(seg_dir):  # 不存在则创建
                os.makedirs(seg_dir)
            file_list = os.listdir(class_path)
            for file_path in file_list:
                fullname = class_path + file_path
                content = self.readfile(fullname).strip()
                content = content.replace("\r\n", " ").strip()
                content_seg = jieba.cut(content)
                self.savefile(seg_dir + file_path, " ".join(content_seg))

    # 分词后的文本信息转化为文本向量信息并对象化
    # Bunch 类提供了一种key，value的对象形式
    # target_name 所有分类集的名称列表
    # label 每个文件的分类标签列表
    # filenames 文件路径
    # contents 分词后文件词向量形式
    def word2vec(self, seg_path, wordbag_path):
        bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
        catelist = os.listdir(seg_path)
        bunch.target_name.extend(catelist)
        for mydir in catelist:
            class_path = seg_path + mydir + "/"
            file_list = os.listdir(class_path)
            for file_path in file_list:
                fullname = class_path + file_path
                bunch.label.append(mydir)
                bunch.filenames.append(fullname)
                bunch.contents.append(self.readfile(fullname).strip())
        # Bunch对象持久化
        file_obj = open(wordbag_path, "wb")
        pickle.dump(bunch, file_obj)
        file_obj.close()

    def readbunchobj(self, path):
        file_obj = open(path, "rb")
        bunch = pickle.load(file_obj)
        file_obj.close()
        return bunch

    def writebunchobj(self, path, bunch_obj):
        file_obj = open(path, "wb")
        pickle.dump(bunch_obj, file_obj)
        file_obj.close()

    # TF-IDF
    def TF_IDF(self, path, space_path):
        bunch = self.readbunchobj(path)
        stpwdlist = self.readfile(self.stopword_path).splitlines()
        # 构建TF-IDF词向量空间对象
        tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                           vocabular={})
        # 使用TfidVectorizer初始化向量空间模型
        vectorizer = TfidfVectorizer(stop_words=stpwdlist, sublinear_tf=True, max_df=0.5)
        # 统计每个词句的TF-IDF值
        transfomer = TfidfTransformer()

        # 文本转为词频矩阵，单独保存字典文件
        # tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
        vectorizer.fit(bunch.contents)
        tfidfspace.tdm = vectorizer.transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
        # 创建词袋的持久化
        self.writebunchobj(space_path, tfidfspace)

    def test_classify(self):
        trainpath = self.space_path_train
        train_set = self.readbunchobj(trainpath)
        testpath = self.space_path_test
        test_set = self.readbunchobj(testpath)
        # 应用贝叶斯
        # alpha=0.001，alpha越小，迭代次数越多，精度越高
        clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)
        # 预测分类结果
        predicted = clf.predict(test_set.tdm)
        total = len(predicted)
        rate = 0
        for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
            if flabel != expct_cate:
                rate += 1
                print(file_name, ": 实际类别：", flabel, "-->预测分类：", expct_cate)
        print("error_rate:", float(rate) * 100 / float(total), "%")

    def metrics_result(self, actual, predict):
        print("精度：{0:.3f}".format(metrics.precision_score(actual, predict)))
        print("召回：{0:0.3f}".format(metrics.recall_score(actual, predict)))
        print("f1-score:{0:.3f}".format(metrics.f1_score(actual, predict)))


if __name__ == '__main__':
    NLP_test = NLP_C()
    # NLP_test.splitwords(NLP_test.corpus_path_test,NLP_test.seg_path_test)
    # NLP_test.splitwords(NLP_test.corpus_path_train,NLP_test.seg_path_train)
    # NLP_test.word2vec(NLP_test.seg_path_test,NLP_test.wordbag_path_test)
    # NLP_test.word2vec(NLP_test.seg_path_train,NLP_test.wordbag_path_train)
    # NLP_test.TF_IDF(NLP_test.wordbag_path_train,NLP_test.space_path_train)
    # NLP_test.TF_IDF(NLP_test.wordbag_path_test,NLP_test.space_path_test)
    # NLP_test.test_classify()



    # seg_list=jieba.cut("小明1995年毕业于清华大学",cut_all=False)
    # print("default Mode:"," ".join(seg_list))
    #
    # seg_list=jieba.cut("小明1995年毕业于清华大学")
    # print(" ".join(seg_list))
    #
    # seg_list=jieba.cut("小明1995年毕业于清华大学",cut_all=True)
    # print("Full Mode:","/ ".join(seg_list))
    #
    # seg_list=jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本东京都大学深造")
    # print("Search enginee Mode:  ","/ ".join(seg_list))
