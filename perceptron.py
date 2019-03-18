import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt




def makePLAData(w,b, numlines):
    w = np.array(w)
    numFeatures = len(w)
    x = np.random.rand(numlines, numFeatures) * 50
    #随机产生numlines个数据的数据集
    cls = np.sign(np.sum(w*x,axis=1)+b)
    #用标准线 w*x+b=0进行分类
    dataSet = np.column_stack((x,cls))
    #样例数据生成

    #存储标准分类线
    x = np.linspace(0, 50, 1000)
    #创建分类线上的点，以点构线。
    y = -w[...,0] / w[...,1] * x - b / w[...,1]
    rows = np.column_stack((x.T, y.T, np.zeros((1000, 1))))
    dataSet = np.row_stack((dataSet, rows))

    return dataSet




def showFigure(dataSet):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    #图例设置
    labels = ['classOne', 'standarLine', 'classTwo', 'modelLine']
    markers = ['o','.','x','.']
    colors = ['r','y','b','g']
    for i in range(4):
        idx = np.where(dataSet[:,2]==i-1)
    #找出同类型的点，返回索引值
        ax.scatter(dataSet[idx, 0], dataSet[idx, 1], marker=markers[i], color=colors[i], label=labels[i], s=10)
    plt.legend(loc = 'upper right')
    plt.show()




def PLA_train(dataSet,plot = False):
    numLines = dataSet.shape[0]
    numFeatures = dataSet.shape[1]
    #模型初始化
    w = np.ones((1, numFeatures-1))
    b = 0.1
    k = 1
    i = 0
    #用梯度下降方法，逐渐调整w和b的值

    while i<numLines:
        if dataSet[i][-1] * (np.sum(w * dataSet[i,0:-1],)+ b) <0:
            #y[i](w*x[i]+b)<0
            w = w + k*dataSet[i][-1] * dataSet[i,0:-1]
            #w = w + k*y[i]
            b = b + k*dataSet[i][-1]
            # b = b + k*y[i]
            i =0
        else:
            i +=1

    x = np.linspace(0,50,1000)
    #创建分类线上的点，以点构线
    y = -w[0][0]/w[0][1]*x - b/w[0][1]
    rows = np.column_stack((x.T,y.T,2*np.ones((1000,1))))
    dataSet = np.row_stack((dataSet,rows))

    showFigure(dataSet)
    return w, b

#测试：
dataSet = makePLAData([1,-2],7,200)
showFigure(dataSet)
w,b= PLA_train(dataSet,True)
  

