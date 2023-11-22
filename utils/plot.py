import matplotlib.pyplot as plt
import seaborn as sns


def plot_futures(futures_dataset):  # 画出前50个重要特征
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.tight_layout()
    sns.set(rc={"figure.figsize": (50, 8)})
    sns.despine(bottom=True)
    sns.barplot(futures_dataset["特征"][:50], futures_dataset["特征重要性值"[:50]])
    plt.ylabel("importance", fontsize=10)
    plt.xlabel("futures", fontsize=10)
    plt.xticks(size=5)  # 由于化合物的分子描述符过长，设置字体较小
    plt.show()