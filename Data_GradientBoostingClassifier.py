import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

data3=pd.read_csv("C:/Users/lengxingcb/Desktop/机器学习算法专项/train_dataset_Y2.csv")
futures=pd.read_csv("data_futures.csv")  #先用预处理过的504个特征进行分类模型训练

data_Y2_pridicts=pd.DataFrame()

SMILES=[]
for i in data3.loc[:,"SMILES"]:
    SMILES.append(i)
data_Y2_pridicts.insert(loc=0,column="SMILES",value=SMILES)

data3=data3.drop(labels=["SMILES"],axis=1) #分离出每个分类项
data_Caco2=data3["Caco-2"]
data_CYP3A4=data3["CYP3A4"]
data_hERG=data3["hERG"]
data_HOB=data3["HOB"]
data_MN=data3["MN"]

test_data_X=pd.read_csv("test_data_X.csv")

def GBC(futureSet):
    futures=futureSet
    gbc_1=GradientBoostingClassifier(random_state=10)
    futures_train, futures_test, target_train, target_test = train_test_split(futures, data_Caco2, test_size=0.2,random_state=0)
    gbc_1.fit(futures_train,target_train)
    train_score=gbc_1.score(futures_train,target_train)
    test_score=gbc_1.score(futures_test,target_test)
    #print("train_Score:",train_score)
    print("Caco-2:",test_score)
    predict_1=pd.DataFrame(gbc_1.predict(test_data_X))
    data_Y2_pridicts.insert(loc=1, column="Cac0-2", value=predict_1)

    gbc_2=GradientBoostingClassifier(random_state=10)
    futures_train, futures_test, target_train, target_test = train_test_split(futures, data_CYP3A4, test_size=0.2,random_state=0)
    gbc_2.fit(futures_train,target_train)
    train_score=gbc_2.score(futures_train,target_train)
    test_score=gbc_2.score(futures_test,target_test)
    #print("train_Score:",train_score)
    print("CYP3A4:",test_score)
    predict_2=pd.DataFrame(gbc_2.predict(test_data_X))
    data_Y2_pridicts.insert(loc=2, column="CYP3A4", value=predict_2)

    gbc_3=GradientBoostingClassifier(random_state=10)
    futures_train, futures_test, target_train, target_test = train_test_split(futures, data_hERG, test_size=0.2,random_state=0)
    gbc_3.fit(futures_train,target_train)
    train_score=gbc_3.score(futures_train,target_train)
    test_score=gbc_3.score(futures_test,target_test)
    #print("train_Score:",train_score)
    print("hERG:",test_score)
    predict_3=pd.DataFrame(gbc_3.predict(test_data_X))
    data_Y2_pridicts.insert(loc=3, column="hERG", value=predict_3)

    gbc_4=GradientBoostingClassifier(random_state=10)
    futures_train, futures_test, target_train, target_test = train_test_split(futures, data_HOB, test_size=0.2,random_state=0)
    gbc_4.fit(futures_train,target_train)
    train_score=gbc_4.score(futures_train,target_train)
    test_score=gbc_4.score(futures_test,target_test)
    #print("train_Score:",train_score)
    print("HOB:",test_score)
    predict_4=pd.DataFrame(gbc_4.predict(test_data_X))
    data_Y2_pridicts.insert(loc=4, column="HOB", value=predict_4)

    gbc_5=GradientBoostingClassifier(random_state=10)
    futures_train, futures_test, target_train, target_test = train_test_split(futures, data_MN, test_size=0.2,random_state=0)
    gbc_5.fit(futures_train,target_train)
    train_score=gbc_5.score(futures_train,target_train)
    test_score=gbc_5.score(futures_test,target_test)
    #print("train_Score:",train_score)
    print("MN:",test_score)
    predict_5=pd.DataFrame(gbc_5.predict(test_data_X))
    data_Y2_pridicts.insert(loc=5, column="MN", value=predict_5)

    data_Y2_pridicts.to_csv("data_Y2_GBC_predicts.csv")
GBC(futures)


new_Dataset=pd.read_csv("new_dataset.csv")
new_Dataset.drop(labels=["SMILES","pIC50"],axis=1,inplace=True)
indexs_2=[]
for i in new_Dataset.columns:
    indexs_2.append(i)
print(indexs_2)
test_data_X=test_data_X.loc[:,indexs_2]

GBC(new_Dataset)