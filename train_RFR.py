import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

from utils.preprocess import preprocess
import pickle


def train(args):
    feature_path = args.feature_path
    target_path = args.target_path
    test_size = args.test_size
    # 预处理部分

    feature, target = preprocess(feature_path=feature_path, target_path=target_path)

    # print(len(data.columns))

    # 预测部分

    futures_train, futures_test, target_train, target_test = train_test_split(feature, target, test_size=test_size,
                                                                              random_state=100)

    with open(args.cfg, 'r') as file:
        rf_params = json.load(file)

    rf = RandomForestRegressor(n_estimators=rf_params['n_estimators'],
                               max_depth=rf_params['max_depth'],
                               min_samples_split=rf_params['min_samples_split'],
                               min_samples_leaf=rf_params['min_samples_leaf'],
                               bootstrap=rf_params['bootstrap'],
                               random_state=rf_params['random_state'])

    rf.fit(futures_train, target_train)

    predict = rf.predict(futures_train)
    print("训练准确度为：:", r2_score(target_train, predict))
    predict = rf.predict(futures_test)
    print("测试准确度为:", r2_score(target_test, predict))

    if args.save:
        # 保存模型到文件
        with open('random_forest_Reg_model.pkl', 'wb') as f:
            pickle.dump(rf, f)

    # 加载模型
    # with open('random_forest_model.pkl', 'rb') as f:
    #     loaded_model = pickle.load(f)

    feature_X = feature.columns
    feature_importances = rf.feature_importances_
    feature_dataset = pd.DataFrame({"特征": feature_X, "特征重要性值": feature_importances})

    feature_dataset.sort_values("特征重要性值", inplace=True, ascending=False)  # 按特征重要性降序排列
    # 是否保存特征重要性到csv
    if args.save_feature_importance:
        feature_dataset.to_csv("feature_importances.csv", index=False)

    # 网格搜索
    if args.GridSearch:
        cfg = {
               "max_depth": [10, 20],
               'min_samples_leaf': [1,2],
               'n_estimators': [50, 100], }

        data_sorted_futures_name = feature_dataset['特征'][:50]
        data_sorted_futures=feature.loc[:,data_sorted_futures_name]
        rf_parameter = [cfg]

        rf = RandomForestRegressor()
        rf_grid = GridSearchCV(rf, rf_parameter, verbose=1)
        futures_train, futures_test, target_train, target_test = train_test_split(data_sorted_futures, target,
                                                                                  test_size=0.2, random_state=100)
        rf_grid.fit(futures_train, target_train)
        print("最优参数为：", rf_grid.best_params_)
        print("训练准确度为：", rf_grid.best_estimator_.score(futures_train, target_train))
        print("测试准确度为:", rf_grid.best_estimator_.score(futures_test, target_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_path",
                        default=r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\train_dataset_X.csv",
                        help='feature.csv')
    parser.add_argument("--target_path",
                        default=r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\train_dataset_Y.csv",
                        help='target.csv')

    parser.add_argument("--save", default=False, help="if save model")

    parser.add_argument("--save_feature_importance", default=False, help="if save feature_importance to csv")

    parser.add_argument("--GridSearch", default=True, help="GridSearch")
    parser.add_argument("--test_size", default=0.2, help="Training test set segmentation ratio ")
    parser.add_argument("--cfg", default="./rf.json", help="random forest config file path")

    args = parser.parse_args()
    train(args)
