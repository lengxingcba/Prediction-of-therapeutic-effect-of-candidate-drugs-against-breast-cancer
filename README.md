# Prediction-of-therapeutic-effect-of-candidate-drugs-against-breast-cancer

抗乳腺癌候选药物治疗效果预测
本课题研究不同药物对乳腺癌的治疗效果，即由不同特征组合而成的化合物对乳腺癌治疗靶标ERα的生物活性。
1.课题采集了1579种化合物对ERα的生物活性数据，存放在“train_dataset_Y.csv”文件中。数据集的第一列提供了化合物的结构式，第二列为化合物对ERα的生物活性值pIC50，值越大表明生物活性越高。
2.上述1579种化合物的729个分子描述符信息（即自变量）存放于“train_dataset_X.csv”文件中（化合物顺序与“train_dataset_Y.csv”一致），每列代表化合物的一个分子描述符。
3.“test_dataset_X.csv”给出了395种化合物的分子描述符信息用于测试。
4.上述1579种化合物的ADMET性质（Absorption吸收、Distribution分布、Metabolism代谢、Excretion排泄、Toxicity毒性）数据存放于“train_dataset_Y2.csv”文件中。该表第一列为化合物的结构式（编号顺序与前面一样），其后5列分别对应每个化合物的ADMET性质，均为布尔值。
任务1. 根据文件“train_dataset_X.csv”和“train_dataset_Y.csv”提供的数据，针对1579种化合物的729个分子描述符进行变量选择，根据变量对生物活性影响的重要性进行排序，并给出前50个对生物活性最具有显著影响的分子描述符（即自变量），并请详细说明分子描述符筛选过程及其合理性。
任务2. 请结合问题1，选择不超过50个分子描述符变量，构建化合物对ERα生物活性的定量预测模型，请叙述建模过程。然后使用构建的预测模型，对文件“test_dataset_X.csv”中的395种化合物的pIC50值进行预测，并通过在线文档提交并检验预测精度。
任务3. 请利用文件“train_dataset_X.csv”提供的729个分子描述符，针对文件“train_dataset_Y2.csv”中提供的1579种化合物的ADMET数据，分别构建化合物的Caco-2、CYP3A4、hERG、HOB、MN的分类预测模型，并简要叙述建模过程。然后使用所构建的5个分类预测模型，对文件“test_dataset_X.csv”中的395种化合物进行相应的预测，并给出预测结果。
