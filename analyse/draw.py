
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def draw_box(data_list_1, data_list_2, y_name, save_path):
    '''
    data_list_1: error_feature_list
    '''
    fig, ax = plt.subplots()
    bp = ax.boxplot(
        [data_list_1, data_list_2],
        labels=['Error', 'Correct'],
        showmeans=True,     # 显示均值
        meanline=True,      # 均值用线表示
        patch_artist=True   # 允许给箱子填充颜色
    )

    # 设置箱子颜色：data_list_1 红色，data_list_2 绿色
    colors = ['red', 'green']
    for box, c in zip(bp['boxes'], colors):
        box.set(facecolor=c, alpha=0.5)

    # 中位数线条样式
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    # 均值线条样式
    for mean in bp['means']:
        mean.set(color='blue', linestyle='--', linewidth=2)

    # 图例：中位数和均值
    legend_lines = [
        Line2D([0], [0], color='black', lw=2),
        Line2D([0], [0], color='blue', lw=2, linestyle='--')
    ]
    ax.legend(legend_lines, ['median', 'mean'])

    ax.set_ylabel(y_name)
    ax.set_title('box plot(red=Error,greed=Correct)')

    plt.tight_layout()
    plt.savefig(save_path)