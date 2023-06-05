# import pymysql
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# def selectSQL():
#     db = pymysql.connect(host="localhost", user="root", password="123456",database="test1", port=3306, charset='utf8')
#     cursor = db.cursor()
#     try:
#         sql = "select * from new_pump_model"
#         cursor.execute(sql)
#         # 这句是把查询结果转化为一个二位列表
#         data = cursor.fetchall()
#         return data
#     except:
#         print("selsct Error")
#
#     db.close()
#
#
# def draw(data):
#     id_list = []
#     data_list = []
#     name_list = []
#     i = 0
# # 这里说一下，data的信息是表里边全部的信息，根据需要取即可
#
#     for item in data:
#     # item 就是表内一行数据，用下标读取第几列即可，从 0 开始
#         id_list.append(item[0])
#         data_list.append(item[7])
#         name_list.append(item[8])
#             # 我这里表里的内容比较多，控制一下数据的个数，可以删除
#         if (i == 432):
#             break
#         else:
#             i = i + 1
#     print(id_list)
#     print(data_list)
#     print(name_list)
#
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     line1, = ax1.plot(id_list, data_list, '-',color='r',linewidth = 2.0,label = 'o' )
#     line2, = ax2.plot(id_list, name_list, '-', color='g', linewidth=2.0)
#     # line2, = ax2.scatter(id_list, name_list, '-', color='g', linewidth=2.0)
#     # plt.legend((line1,line2),('oilFlow','inletoilpres'),frameon=False,loc="lower right",fontsize='small')
#     # plt.legend((line1, line2), ('oilFlow', 'tag'), frameon=False, loc="lower right", fontsize='small')
#     plt.legend((line1, line2), ('inletoilpres', 'tag'), frameon=False, loc="center right", fontsize='small')
#     plt.xticks(range(0, 431, 25))
#     # plt.yticks(np.linspace(0, 0.35, 0.05))
#     ax1.set_title('model')
#     ax1.set_xlabel('serial',fontsize =14)
#     # ax1.set_ylabel('oilFlow',fontsize = 14)
#     ax1.set_ylabel('inletoilpres', fontsize=14)
#     # ax2.set_ylabel('inletoilpres', fontsize=14)
#     ax2.set_ylabel('tag', fontsize=14)
#     plt.grid(False)
#     plt.tick_params(direction='in')
#     # plt.savefig("D:/python_test/model/oilflow.svg")
#     plt.show()
#
# def main():
#     # 先查出数据，然后再绘图
#     draw(selectSQL())
#
#
# if __name__ == "__main__":
#     main()

#2017-11-10断泵试验进油压力与标记的图像
# #




#
#
# import pandas as pd
# import numpy as np
# import  pymysql
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import  classification_report
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_curve,auc
# from sklearn.metrics import confusion_matrix
# import  matplotlib.pyplot as plt
#
# #连接数据库
# conn = pymysql.connect(host="localhost", user="root", password="123456",database="test1", port=3306, charset='utf8')
# # 读取数据
# data = pd.read_sql_query("SELECT oilFlow , inletOilPres, tag FROM new_pump_model" ,conn)
#
#
# # 划分训练集和测试集
# X = data.drop('tag',axis =1)
# y = data['tag']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=42)
#
# # 初始化逻辑回归模型训练
# model = LogisticRegression()
# #训练模型
# model.fit(X_train, y_train)
# predict_tag = model.predict(X_test)
# # score = model.score(X_train, y_train)
# score = model.score(X_test, y_test)
# print("准确率:",score)
#
# coef = model.coef_
# intercept = model.intercept_
#
# print('模型系数为：',coef)
# print('模型截距为：',intercept)
# print(classification_report(y_test,model.predict(X_test)))
#
# param_grid = {'C':[0.1,1,20],'solver':['lbfgs','liblinear']}
# #上面是定义超参数的范围
# grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=5)
# #初始化GridSearchCV对象
# grid_search.fit(X_train,y_train)
# #对模型进行训练和调整
# print('最优的参数为：',grid_search.best_params_)
# print('最优的准确率：',grid_search.best_score_)
#
# y_score = model.predict_proba(X_test)[:,1]
# fpr , tpr, thresholds = roc_curve(y_test,y_score)
# a = pd.DataFrame()
# a['阈值'] = list(thresholds)
# a['假报警率'] = list(fpr)
# a['命中率'] = list(tpr)
# print(a)
#
#
# roc_auc = auc(fpr,tpr)
# #计算ROC曲线和AUC指标
# print('AUG值：',roc_auc)
#
# plt.figure()
# plt.plot(fpr,tpr,color = 'blue',lw = 2,label = 'ROC curve (area = %0.2f)'% roc_auc)
# plt.plot([0,1],[0,1],color = 'navy' ,lw = 2, linestyle = '--')
# plt.xlim([0.0,1.0])
# plt.ylim([0.0,1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc = 'lower right')
# plt.show()

#2017-11-10断泵试验建立断泵故障数据模型






import pymysql
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def selectSQL():
    db = pymysql.connect(host="localhost", user="root", password="123456",database="test1", port=3306, charset='utf8')
    cursor = db.cursor()
    try:
        sql = "select oilFlow,inletoilPres from new_pump_model"
        cursor.execute(sql)
        # 这句是把查询结果转化为一个二位列表
        data = cursor.fetchall()
        return data
    except:
        print("selsct Error")

    db.close()


def draw(data):
    # id_list = []
    oilFlow_data = [row[0] for row in data]
    inletoilPres_data = [row[1] for row in data]
    # data_list = []
    # name_list = []
    print(oilFlow_data)
    print(inletoilPres_data)

    slopes_oilFlow = []
    for i in range(21,len(oilFlow_data)):
        slope1 = (oilFlow_data[i]-oilFlow_data[i-20])/ 20
        slopes_oilFlow.append(slope1)

    slopes_inletoilPres = []
    for i in range(21, len(inletoilPres_data)):
        slope2 = (inletoilPres_data[i] - inletoilPres_data[i-20])/ 20
        slopes_inletoilPres.append(slope2)
        if (i == 432):
            break
        else:
            i = i + 1
    # print(id_list)

    print(slopes_oilFlow)
    print(slopes_inletoilPres)



    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twinx()
    line1, = ax1.plot( slopes_oilFlow, '-',color='r',linewidth = 2.0,label = 'oilFlow' )
    line2, = ax2.plot( slopes_inletoilPres, '-', color='b', linewidth=2.0,label = 'inletoilPres')
    # # line2, = ax2.scatter(id_list, name_list, '-', color='g', linewidth=2.0)
    plt.legend((line1,line2),('oilFlow','inletoilpres'),frameon=False,loc="lower right",fontsize='small')
    ax1.set_title('change rate with an interval of 20')
    ax1.set_xlabel('serial',fontsize =14)
    ax1.set_ylabel('oilFlow',fontsize = 14)
    ax2.set_ylabel('inletoilpres', fontsize=12)
    plt.show()

def main():
    # 先查出数据，然后再绘图
    draw(selectSQL())


if __name__ == "__main__":
    main()

#017-11-10断泵试验油流量、进油压力斜率折线图





# import pandas as pd
# import numpy as np
# import  pymysql
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.model_selection import cross_val_score
# # from sklearn.metrics import roc_curve,auc
# from sklearn.metrics import confusion_matrix
# import  matplotlib.pyplot as plt
#
# #连接数据库
# conn = pymysql.connect(host="localhost", user="root", password="123456",database="test1", port=3306, charset='utf8')
# # 读取数据表new_pump_model
# data = pd.read_sql_query("SELECT oilFlow , inletOilPres, tag FROM new_pump_model" ,conn)
#
#
# # 将油流量和进油压力组合成一个特征向量
# X =np.array(data[['oilFlow','inletOilPres']])
# y = np.array(data['tag'])
# # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=42)
#
# # 初始化逻辑回归模型训练
# model = LogisticRegression()
# #训练模型
# model.fit(X, y)
# # 读取数据表determine_pump_model
# data_test = pd.read_sql_query("SELECT oilFlow , inletOilPres, tag FROM determine_pump_model" ,conn)
# X_test = np.array(data_test[['oilFlow','inletOilPres']])
# y_pred = model.predict(X_test)
# print('预测结果为：',y_pred)
# y_true = np.array(data_test['tag'])
# #计算混淆矩阵
# cm = confusion_matrix(y_true,y_pred)
# print('Confusion Matrix:\n',cm)
# #计算准确率、精确率、和召回率
# accuracy = accuracy_score(y_true,y_pred)
# precision = precision_score(y_true,y_pred)
# recall = recall_score(y_true,y_pred)
# print('准确率:',accuracy)
# print('精确率:',precision)
# print('召回率:',recall)
# #使用交叉验证评估模型性能，假设使用5折交叉验证
# scores = cross_val_score(model,X,y,cv=5)
# print('Cross validation Scores:',scores)
# print('Mean Score:',np.mean(scores))
# #对模型进行调整，假设调整参数C和penalty
# model_tuned = LogisticRegression(C =0.1,penalty='l1')
# model_tuned.fit(X,y)
# #使用交叉验证评估调整后的模型性能
# scores_tuned = cross_val_score(model_tuned,X,y,cv=5)
# print('Cross validation Scores(Tuned Model):',scores)
# print('Mean Score(Tuned Model):',np.mean(scores))
# y1_pred = model_tuned.predict(X_test)
# y1_true =np.array(data_test['tag'])
#
# cm1 = confusion_matrix(y1_true,y1_pred)
# print('Confusion Matrix(Tuned Model):\n',cm)
#
# accuracy1 = accuracy_score(y1_true,y1_pred)
# precision1 = precision_score(y1_true,y1_pred)
# recall1 = recall_score(y1_true,y1_pred)
# print('准确率(交叉验证):',accuracy1)
# print('精确率(交叉验证):',precision1)
# print('召回率(交叉验证):',recall1)
# ##判断模型








































