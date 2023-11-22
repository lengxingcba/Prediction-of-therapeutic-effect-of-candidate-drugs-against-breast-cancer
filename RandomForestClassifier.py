import argparse
import json
import os
import pickle
from utils.preprocess import preprocess
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np

# RandomForest
def RFClassifier(cfg, features, target, name, save=True):
    with open(cfg, 'r') as file:
        rf_params = json.load(file)

    rfc = RandomForestClassifier(n_estimators=rf_params['n_estimators'],
                                 criterion=rf_params['criterion'],
                                 max_depth=rf_params['max_depth'],
                                 min_samples_split=rf_params['min_samples_split'],
                                 min_samples_leaf=rf_params['min_samples_leaf'],
                                 bootstrap=rf_params['bootstrap'],
                                 random_state=rf_params['random_state'])

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,
                                                                                random_state=0)
    rfc.fit(features_train, target_train)
    sc = rfc.score(features_test, target_test)
    if save:
        # 保存模型到文件
        if not os.path.exists("./model"):
            os.makedirs("./model")
        save_path = "./model/random_forest_classify_model_{}.pkl".format(name)
        # assert os.path.exists(save_path), 'model {} is exists'.format(save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(rfc, f)

    return sc

# GradientBoosting
def GBClassifier(cfg, features, target, name, save=True):
    with open(cfg, 'r') as file:
        rf_params = json.load(file)

    rfc = GradientBoostingClassifier(n_estimators=rf_params['n_estimators'],
                                     criterion=rf_params['criterion'],
                                     max_depth=rf_params['max_depth'],
                                     min_samples_split=rf_params['min_samples_split'],
                                     min_samples_leaf=rf_params['min_samples_leaf'],
                                     random_state=rf_params['random_state'])

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,
                                                                                random_state=0)
    rfc.fit(features_train, target_train)
    sc = rfc.score(features_test, target_test)
    if save:
        # 保存模型到文件
        if not os.path.exists("./model"):
            os.makedirs("./model")
        save_path = "./model/gradient_boosting_classify_model_{}.pkl".format(name)
        # assert os.path.exists(save_path), 'model {} is exists'.format(save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(rfc, f)

    return sc


def train(args):
    feature_path = args.feature_path
    target_path = args.target_path
    cfg = args.cfg

    feature, target = preprocess(feature_path, target_path, save=False)
    print(feature)

    if args.GridSearch:
        print("GridSearch start")
        cfg = {
            "max_depth": [10, 20],
            'min_samples_leaf': [1, 2],
            'n_estimators': [50, 100], }
        rf_parameter = [cfg]

        for i in range(len(target.columns)):
            rf = RandomForestClassifier()
            rf_grid = GridSearchCV(rf, rf_parameter, verbose=1)
            features_train, features_test, target_train, target_test = train_test_split(feature, target.iloc[:, i],
                                                                                        test_size=0.2, random_state=100)
            rf_grid.fit(features_train, target_train)
            print("对{}的分类最优参数为：".format(target.columns[i]), rf_grid.best_params_)
            print("对{}的最好训练准确度为：".format(target.columns[i]),
                  rf_grid.best_estimator_.score(features_train, target_train))
            print("对{}的最好测试准确度为:".format(target.columns[i]),
                  rf_grid.best_estimator_.score(features_test, target_test))

            if args.save_cfg:
                with open("cfg_rfc_{}.json".format(target.columns[i]), 'w') as w:
                    best_params = rf_grid.best_params_
                    json.dump(best_params, w)

    else:

        for i in range(len(target.columns)):
            sc = RFClassifier(cfg, feature, target.iloc[:, i], target.columns[i], save=args.save)
            print("acc_{} :".format(target.columns[i]), sc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_path",
                        default=r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\train_dataset_X.csv",
                        help='feature.csv')
    parser.add_argument("--target_path",
                        default=r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\train_dataset_Y2.csv",
                        help='target.csv')

    parser.add_argument("--save", default=True, help="if save model")

    # parser.add_argument("--save_feature_importance", default=False, help="if save feature_importance to csv")

    parser.add_argument("--GridSearch", default=False, help="GridSearch")
    parser.add_argument("--save_cfg", default=True, help="if save GridSearch best model cfg")
    # parser.add_argument("--test_size", default=0.2, help="Training test set segmentation ratio ")
    parser.add_argument("--cfg", default="./rfc.json", help="random forest config file path")

    args = parser.parse_args()
    train(args)
