import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def preprocess(feature_path: str, target_path: str, save=False):
    """
    去除全为0的列
    :param feature_path: feature.sv
    :param target_path: target.csv
    :param save: if save result to csv file
    :return: if save=True preprocess.csv | feature,target
    """
    data = pd.read_csv(feature_path)
    data = data.loc[:, (data != 0).any(axis=0)]
    # 0值填充
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    imputer.fit_transform(data)

    feature = data.iloc[:, 1:]
    feature_names = feature.columns
    with open("sorted_feature.txt", 'w') as s:
        s.writelines(feature_name + "\n" for feature_name in feature_names)

    target = pd.read_csv(target_path).iloc[:, 1:]
    if save:
        data_concat = pd.concat([data, target], axis=1)
        data_concat.to_csv('preprocess.csv', index=False)
    else:
        return feature, target


def predict_process(data: str, columns_sample, features=None):
    """
    :param data: csv_path
    :param feature: feature.txt
    :return: dataframe
    """
    data = pd.read_csv(data)



    samples = data[columns_sample]
    if features is not None:
        with open(features, 'r') as f:
            columns = [line.strip() for line in f]
        data = data[columns]
    else:
        data = data.iloc[:, 1:]
        data = data.loc[:, (data != 0).any(axis=0)]
        # 0值填充
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        imputer.fit_transform(data)
    return data, samples

# data=predict_process(r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\test_dataset_X.csv",feature='./sorted_feature.txt')
