import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MultipleLocator
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1-accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20,12))
    # myCm = cm[0][:]
    # print('myCm is ',myCm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        sumCm = cm.sum(axis=1)[:,np.newaxis]
        for i in range(cm.shape[0]):
            if sumCm[i][0]!=0:
                cm[i][:] = cm[i][:]/sumCm[i][0]
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def plot_confusion_matrix2(cm, labels,title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    #save
    plt.savefig(savname)


def main():
    ave_preds = np.arange(101)
    ave_preds = ave_preds.astype(np.int)
    gt = np.arange(0,101)
    gt = gt.astype(np.int)
    print(ave_preds)
    print(gt)
    labels = np.arange(0,101)

    # cm = confusion_matrix(ave_preds, gt,labels=np.arange(0,101))
    # # plotCM(labels,cm,'./res.jpg')
    # # print("cm is ",cm)
    # #
    # plot_confusion_matrix(cm, title='Confusion matrix',target_names=labels)
    # plt.show()
    import pandas as pd
    df = pd.crosstab(gt,ave_preds)
    #df.to_csv('./a.csv',sep=',',header=True,index=True)
    trueLabel = df.index.values
    predLabel = df.columns.values
    cm = df.values
    cm = cm/cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print('trueLabel is ',trueLabel)
    print('predLabel is ',predLabel)

    xLabel = [trueLabel[0]]
    yLabel = predLabel
    data = cm[0,:].reshape(len(predLabel),1)
    print('data is ',data)
    #fig = plt.figure(figsize=(50,20))
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    #ax = fig.add_subplot(999)
    fig,ax = plt.subplots(figsize=(30,20))
    # 定义横纵坐标的刻度
    ax.set_yticks(range(yLabel.shape[0]))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.hot_r,aspect=0.2)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title("This is a title of "+str(xLabel[0]))
    plt.show()
    # show



if __name__ == '__main__':
    main()