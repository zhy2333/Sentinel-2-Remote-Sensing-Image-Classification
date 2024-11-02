import sys
import os
from PyQt5 import QtWidgets, QtGui
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from functools import partial
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel,QGridLayout,QComboBox
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QGridLayout, QTextEdit,QProgressBar
from PyQt5.QtGui import QPalette, QBrush, QPixmap

class ImageClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('遥感图像分类器')
        self.resize(1500, 1000)

        # 垂直布局
        grid = QGridLayout()
        grid.setSpacing(10)
        # Layout


        # 第一栏：导入图像路径

        self.imgPathText = QtWidgets.QLineEdit(self)
        self.imgPathBtn = QtWidgets.QPushButton("导入哨兵二号图像路径", self)
        self.imgPathBtn.clicked.connect(self.loadImagePath)

        # 第二栏：导入标签路径

        self.labelPathText = QtWidgets.QLineEdit(self)
        self.labelPathBtn = QtWidgets.QPushButton("导入标签路径", self)
        self.labelPathBtn.clicked.connect(self.loadLabelPath)
        self.nInput = QLineEdit(self)
        self.nInput.setPlaceholderText("请输入地物数量 (例如 10)")
        # 标记样本图像按钮
        self.showSampleBtn = QtWidgets.QPushButton("地物数量", self)
        self.showSampleBtn.clicked.connect(self.showSampleImage)
        #显示标记样本图像
        self.showSampleBtn1 = QtWidgets.QPushButton("显示标记样本图像", self)
        self.showSampleBtn1.clicked.connect(self.openimage)
        # 显示分类图像按钮
        self.showClassifyBtn = QtWidgets.QPushButton("分类", self)
        self.showClassifyBtn.clicked.connect(self.showClassifyImage)
        self.showClassifyBtn1 = QtWidgets.QPushButton("显示分类图像", self)
        self.showClassifyBtn1.clicked.connect(self.openimage1)
        #进度条
        self.progressBar = QProgressBar(self)

        self.progressBar.setValue(0)  # 初始值为0
        self.progressBar1 = QProgressBar(self)

        self.progressBar1.setValue(0)  # 初始值为0
        # 图像显示部分
        self.lb1 = QtWidgets.QLabel(self)

        self.lb1.setStyleSheet("border: 1px solid red")# 用于显示标记样本图像
        self.lb2 = QtWidgets.QLabel(self)  # 用于显示分类图像
        self.lb2.setStyleSheet("border: 1px solid red")
        # Add widgets to layout
        self.sampleCountsTextEdit = QTextEdit(self)
        self.sampleCountsTextEdit.setReadOnly(True)

        self.groupedSampleCountsTextEdit = QTextEdit(self)
        self.groupedSampleCountsTextEdit.setReadOnly(True)
        self.groupedSampleCountsTextEdit.setFixedSize(400, 40)
        self.sampleCountsTextEdit.setFixedSize(400, 40)
        #模型选择
        self.modelComboBox = QComboBox(self)

        self.modelComboBox.addItem("KNN")
        self.modelComboBox.addItem("SVM")
        self.modelComboBox.addItem("Random Forest")
        self.testAccuracyTextEdit = QtWidgets.QTextEdit(self)
        self.testAccuracyTextEdit.setFixedSize(400, 40)
        grid.addWidget(self.imgPathBtn,1,1,1,2)
        grid.addWidget(self.imgPathText, 1, 3, 1, 10)#13
        grid.addWidget(self.labelPathBtn, 2, 1, 1, 2)
        grid.addWidget(self.labelPathText, 2, 3, 1, 10)
        grid.addWidget(QLabel("地物数量："), 3, 1, 1, 2)
        grid.addWidget(self.nInput, 3, 2, 1, 2)
        grid.addWidget(self.showSampleBtn, 4, 1, 1, 1)
        grid.addWidget(self.progressBar, 4, 2, 1, 2)#5
        grid.addWidget(QLabel("选择模型："), 4, 7, 1, 1)
        grid.addWidget(self.modelComboBox, 4, 8, 1, 5)
        grid.addWidget(QLabel("分组前选取样本数目："), 5, 1, 1, 1)
        grid.addWidget(self.sampleCountsTextEdit, 5, 2, 1, 2)
        grid.addWidget(self.showClassifyBtn, 5, 7, 1, 2)
        grid.addWidget(self.progressBar1, 5,9, 1, 4)
        grid.addWidget(QLabel("分组后选取样本数目："), 6, 1, 1, 1)
        grid.addWidget(self.groupedSampleCountsTextEdit, 6, 2, 1, 2)
        grid.addWidget(QLabel("模型分类精度："), 6, 7, 1, 1)
        grid.addWidget(self.testAccuracyTextEdit, 6, 8, 1, 6)
        grid.addWidget(self.showSampleBtn1, 7, 2, 1, 1)
        grid.addWidget(self.showClassifyBtn1, 7, 9, 1, 2)
        grid.addWidget(self.lb1, 8, 1, 4, 3)
        grid.addWidget(self.lb2, 8, 7, 4, 6)

        self.setLayout(grid)

        self.show()


    def openimage(self):
        im_path=self.biaojiimg

        self.showImage=QPixmap(im_path).scaled(450,650)
        self.lb1.setPixmap(self.showImage)
    def openimage1(self):
        im_path=self.classifyimgpath
        print(im_path)
        self.showImage1=QPixmap(im_path).scaled(450,650)
        self.lb2.setPixmap(self.showImage1)
    def loadImagePath(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图像文件", "", "TIF Files (*.tif);;All Files (*)",
                                                            options=options)
        if fileName:
            self.imgPathText.setText(fileName)

    def loadLabelPath(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择标签文件", "", "TIF Files (*.tif);;All Files (*)",
                                                            options=options)
        if fileName:
            self.labelPathText.setText(fileName)

    def showSampleImage(self):
        img_path = self.imgPathText.text()
        label_path = self.labelPathText.text()
        txt_path = r"D:\newsave8.txt"

        # Code from "代码1"
        landset_ds = gdal.Open(img_path)
        label_ds = gdal.Open(label_path)
        Tif_width = landset_ds.RasterXSize
        Tif_height = landset_ds.RasterYSize
        Landset_data = landset_ds.ReadAsArray(0, 0, Tif_width, Tif_height)
        Label_data = label_ds.ReadAsArray(0, 0, Tif_width, Tif_height)
        self.progressBar.setValue(10)
        if os.path.exists(txt_path):
            os.remove(txt_path)

        with open(txt_path, 'w') as file_write_obj:
            Features =int(self.nInput.text()) if self.nInput.text() else 11
            count = [0] * Features
            for j in range(Label_data.shape[0]):
                for k in range(Label_data.shape[1]):
                    label = Label_data[j, k]
                    if 1 <= label <= Features:#如果地物从编号1开始标记
                        new_label = label - 1
                    else:
                        continue
                    count[new_label] += 1
                    var = f"{j},{k},"
                    var += ",".join(map(str, Landset_data[:, j, k]))
                    var += f",{new_label}"
                    file_write_obj.write(var + '\n')
            self.sampleCountsTextEdit.append(f"{','.join(map(str, count))}")
            for i in range(len(count)):
                print(f"count[{i}] = {count[i]}")
        self.progressBar.setValue(30)
        # Continue with "代码2"
        SavePath = r"D:\ClassifyModel5.pickle"
        self.RFpath = SavePath
        iris_label_with_param = partial(self.Iris_label, n=Features)
        data = np.loadtxt(txt_path, dtype=float, delimiter=',', converters={11: iris_label_with_param})
        data1 = np.loadtxt(txt_path, dtype=int, delimiter=',', converters={11: iris_label_with_param})
        data1 = data1.tolist()
        n = len(data1)

        list1 = [0] * Features
        for row in data1:
            x = row[-1]
            list1[x] += 1
        for i in range(len(list1)):
            print(f"list1[{i}] = {list1[i]}")
        list5 = [0] * Features
        data1 = list(map(list, zip(*data)))
        max0, min0 = max(max(data1[i]) for i in range(2, 11)) * 10000, min(min(data1[i]) for i in range(2, 11)) * 10000
        bins_sum = math.sqrt(max0 - min0 + 1)
        print(bins_sum)
        for i in range(Features):
            list5[i] = int(bins_sum * (list1[i] / len(data) + 0.05))
            print(list5[i])
        count = [0] * Features
        m = 0
        sample0 = data[:list1[0]]
        sample0 = self.getFinal(sample0, list5[0], max0)
        print(len(sample0))
        sample_counts = [len(sample0)]
        self.progressBar.setValue(50)
        for i in range(Features-1):
            m += list1[i]
            n = m + list1[i + 1]
            sample_h = data[m:n]
            sample_h = self.getFinal(sample_h, list5[i + 1], max0)
            print(len(sample_h))
            sample_counts.append(len(sample_h))
            sample0 += sample_h
        self.groupedSampleCountsTextEdit.append(f"{','.join(map(str, sample_counts))}")
        x, y = np.split(np.array(sample0), indices_or_sections=(11,), axis=1)
        coords = np.array(sample0)[:, :2]  # 提取行和列坐标
        features = np.array(sample0)[:, 2:11]  # 提取特征
        self.progressBar.setValue(60)
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test, coords_train, coords_test = train_test_split(features, y, coords,
                                                                                       test_size=0.2, random_state=42)


        ds = gdal.Open(img_path)
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        red_band = ds.GetRasterBand(4).ReadAsArray()
        green_band = ds.GetRasterBand(3).ReadAsArray()
        blue_band = ds.GetRasterBand(2).ReadAsArray()
        rgb_image = np.dstack((red_band, green_band, blue_band))
        fig, ax = plt.subplots(figsize=(10, 8))

        # 显示图像
        im = ax.imshow(rgb_image)
        self.progressBar.setValue(70)
        # 添加颜色条带
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=10)  # 设置颜色条带的刻度字体大小

        # 调整子图位置和边界框
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        for sample in coords_train:
            row = sample[0]
            col = sample[1]
            plt.scatter(col, row, c='red', marker='o', s=0.05)
        self.progressBar.setValue(80)
        save_path = "D:/annotated_image5.png"
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1)
        self.biaojiimg=save_path
        plt.savefig(save_path, format='png')
        #plt.show()
        ds = None
        self.progressBar.setValue(100)
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test

        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        # 创建并训练 SVM 分类器



        #QMessageBox.information(self, '完成', '分组选取样本完成', QMessageBox.Ok)

    def showClassifyImage(self):
        model_index = self.modelComboBox.currentIndex()
        if model_index == 0:
            model_name = "KNN"
            model = KNeighborsClassifier(n_neighbors=9)
        elif model_index == 1:
            model_name = "SVM"
            model = SVC(kernel='rbf', gamma='auto')
        elif model_index == 2:
            model_name = "Random Forest"
            model = RandomForestClassifier(n_estimators=100)
        model.fit(self.x_train,self.y_train.ravel())
        self.progressBar1.setValue(10)
        # Continue with "代码3"
        test_predictions = model.predict(self.x_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        print("测试集精度:", test_accuracy)

        test_accuracy_text = f"{test_accuracy * 100:.2f}%"
        self.progressBar1.setValue(30)
        # 更新 QTextEdit 显示测试集精度信息
        self.testAccuracyTextEdit.setText(test_accuracy_text)
        Landset_Path = self.imgPathText.text()
        SavePath = r"D:\saveClassify7.tif"

        dataset = self.readTif(Landset_Path)
        Tif_width = dataset.RasterXSize
        Tif_height = dataset.RasterYSize
        Tif_geotrans = dataset.GetGeoTransform()
        Tif_proj = dataset.GetProjection()
        Landset_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)
        self.progressBar1.setValue(40)

        rf_model = model


        data = np.zeros((Landset_data.shape[0], Landset_data.shape[1] * Landset_data.shape[2]))
        for i in range(Landset_data.shape[0]):
            data[i] = Landset_data[i].flatten()
        data = data.swapaxes(0, 1)
        self.progressBar1.setValue(50)
        pred = rf_model.predict(data)
        pred = pred.reshape(Landset_data.shape[1], Landset_data.shape[2])
        pred = pred.astype(np.uint8)
        self.progressBar1.setValue(60)
        print("222")
        self.progressBar1.setValue(70)
        self.writeTiff(pred, Tif_geotrans, Tif_proj, SavePath)
        print("333")
        self.classifyimgpath=SavePath
        self.progressBar1.setValue(80)
        self.progressBar1.setValue(100)




    def Iris_label(self, s, n):
        # 动态生成 b'数字': 数字 字典，数量由 n 决定
        it = {bytes(str(i), 'utf-8'): i for i in range(n)}
        return it[s]

    def getFinal(self,type00, n, max):
        if n <= 1:
            n = 2
        list2 = []
        num = len(type00)
        bin_list = [[] for _ in range(n)]
        for i in range(len(type00)):
            list2.append(type00[i][5] * 0.299 + type00[i][4] * 0.587 + 0.114 * type00[i][3])
            bin_list[int(list2[i] / ((max + (n - 1)) / n))].append(type00[i])
        list_end = []
        for i in range(n):
            list3 = [random.randint(0, len(bin_list[i]) - 1) for _ in
                     range(int(0.05 * len(bin_list[i]) * len(bin_list[i]) / num))]
            list_end.extend([bin_list[i][j] for j in list3])
        return list_end
    def readTif(self, path):
        dataset = gdal.Open(path)
        if dataset is None:
            print(path + "文件无法打开")
        return dataset




    def writeTiff(self, im_data, im_geotrans, im_proj, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        # 定义 50 个颜色
        color_list = [
            (115, 74, 18),  # 棕色
            (38, 112, 0),  # 深绿色
            (0, 0, 0),  # 黑色
            (255, 250, 240),  # 米色
            (135, 206, 235),  # 天蓝色
            (255, 0, 0),  # 红色
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 蓝色
            (255, 255, 0),  # 黄色
            (128, 0, 0),  # 栗色
            (128, 128, 0),  # 橄榄色
            (0, 128, 0),  # 暗绿色
            (128, 0, 128),  # 紫色
            (0, 128, 128),  # 青色
            (192, 192, 192),  # 银色
            (128, 128, 128),  # 灰色
            (0, 0, 128),  # 海军蓝
            (255, 165, 0),  # 橙色
            (255, 20, 147),  # 深粉色
            (75, 0, 130),  # 靛色
            (255, 192, 203),  # 粉红色
            (255, 255, 255),  # 白色
            (139, 69, 19),  # 巧克力色
            (47, 79, 79),  # 深灰
            (70, 130, 180),  # 钢蓝
            (240, 128, 128),  # 浅珊瑚色
            (34, 139, 34),  # 森林绿
            (255, 215, 0),  # 金色
            (173, 255, 47),  # 黄绿色
            (0, 191, 255),  # 深天蓝
            (138, 43, 226),  # 蓝紫色
            (255, 105, 180),  # 热粉色
            (250, 128, 114),  # 三文鱼色
            (124, 252, 0),  # 草坪绿
            (199, 21, 133),  # 紫红色
            (152, 251, 152),  # 苍白绿色
            (72, 61, 139),  # 深紫色
            (220, 20, 60),  # 猩红色
            (64, 224, 208),  # 绿松石色
            (176, 196, 222),  # 淡蓝色
            (238, 130, 238),  # 紫罗兰
            (144, 238, 144),  # 浅绿色
            (95, 158, 160),  # 青铜色
            (255, 228, 225),  # 薄粉色
            (245, 245, 220),  # 米黄色
            (32, 178, 170),  # 淡青色
            (176, 224, 230),  # 粉蓝色
            (250, 250, 210),  # 浅黄色
            (240, 230, 140),  # 卡其色
            (211, 211, 211),  # 淡灰色
            (100, 149, 237)  # 矢车菊蓝色
        ]

        # 设置颜色数量 n（在实际代码中可以根据需要动态指定）
        n =int(self.nInput.text()) if self.nInput.text() else 11   # 假设要选择前 10 个颜色

        # 创建颜色表并动态添加颜色
        colors = gdal.ColorTable()
        for i in range(n):
            colors.SetColorEntry(i, color_list[i])

        # 将颜色表应用到波段中
        for i in range(im_bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(im_data[i])
            band.SetColorTable(colors)


            # 关闭数据集
        del dataset
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = ImageClassifierApp()
    sys.exit(app.exec_())
 
