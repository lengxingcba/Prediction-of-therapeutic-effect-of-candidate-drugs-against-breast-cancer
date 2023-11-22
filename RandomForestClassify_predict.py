import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from utils.preprocess import predict_process


def predict(args):
    model_path = args.model
    data_path = args.data
    feature, samples = predict_process(data_path, "SMILES",features="./utils/sorted_feature.txt")
    # 加载模型
    # r = pd.DataFrame()
    for model in os.listdir(model_path):
        model_name = model.split(".")[-2].split("_")[-1]
        with open(os.path.join(model_path, model), 'rb') as f:
            loaded_model = pickle.load(f)

        # print(feature.columns)

        pred = loaded_model.predict(feature)

        samples = pd.concat([samples, pd.DataFrame({model_name: pred})],axis=1)

    if args.save:
        samples.to_csv("result_all.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        default=r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\test_dataset_X.csv",
                        help='data')

    parser.add_argument("--model", default="./model", help="model path")
    parser.add_argument("--save", default=True, help="if save result")

    args = parser.parse_args()
    predict(args)
