import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from utils.preprocess import predict_process




def predict(args):
    model_path=args.model
    data_path=args.data
    feature=args.feature
    # 加载模型
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    feature,samples=predict_process(data_path,feature,"SMILES")
    # print(feature.columns)

    result=loaded_model.predict(feature)
    result=pd.DataFrame({"result":result})
    result=pd.concat([samples,result],axis=1)

    if args.save:
        result.to_csv("result.csv",index=False)





if __name__=="__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--data",
                        default=r"C:\Users\lengxingcb\Desktop\抗乳腺癌候选药物治疗效果预测\test_dataset_X.csv",
                        help='data')


    parser.add_argument("--save_feature_importance", default=False, help="if save feature_importance to csv")

    parser.add_argument("--model", default="./random_forest_model.pkl", help="model path")
    parser.add_argument("--feature", default="./utils/sorted_feature.txt", help="model path")
    parser.add_argument("--save", default=True, help="if save result")

    args = parser.parse_args()
    predict(args)







