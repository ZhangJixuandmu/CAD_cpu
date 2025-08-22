#热力图尝试
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

# === 数据读取 ===
base_path = os.getcwd()
file_path = os.path.join(base_path, "data", "data.xlsx")
df = pd.read_excel(file_path, engine="openpyxl")
numeric_df = df.select_dtypes(include=[np.number])  # 只保留数值型变量
cols = numeric_df.columns
n = len(cols)

# === 计算相关系数和 p 值矩阵 ===
corr_matrix = numeric_df.corr()
pval_matrix = pd.DataFrame(np.ones_like(corr_matrix), columns=cols, index=cols)

for i in range(n):
    for j in range(n):
        if i != j:
            corr, p = stats.pearsonr(numeric_df[cols[i]], numeric_df[cols[j]])
            pval_matrix.iloc[i, j] = p

# === 开始绘图 ===
fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n), dpi=300)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
cmap = sns.diverging_palette(120, 0, as_cmap=True)
norm = plt.Normalize(vmin=-1, vmax=1)

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])

        # === 对角线：绘制变量分布 ===
        if i == j:
            sns.histplot(numeric_df[cols[i]], kde=True, ax=ax, color='black', edgecolor=None)
            ax.set_ylabel('')
            ax.set_xlabel('')

        # === 下三角：回归拟合图 ===
        elif i > j:
            sns.regplot(x=cols[j], y=cols[i], data=numeric_df, ax=ax,
                        scatter_kws={'s': 10, 'color': 'red'}, line_kws={'color': 'blue'}, ci=95)

        # === 上三角：绘制色块 + 相关系数文字 + 显著性 ===
        else:
            corr_val = corr_matrix.iloc[i, j]
            p = pval_matrix.iloc[i, j]

            color = cmap(norm(corr_val)) if not np.isnan(corr_val) else 'white'
            ax.set_facecolor(color)

            # 相关系数文字
            ax.text(0.5, 0.6, f"{corr_val:.2f}", ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black', transform=ax.transAxes)

            # 显著性星号
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''
            ax.text(0.5, 0.35, stars, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black', transform=ax.transAxes)

# === 设置坐标轴标签 ===
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == n - 1:  # 最底行加 x 标签
            ax.set_xlabel(cols[j], fontsize=10, rotation=45)
        else:
            ax.set_xlabel('')
        if j == 0:  # 最左列加 y 标签
            ax.set_ylabel(cols[i], fontsize=10, rotation=0, labelpad=35)
        else:
            ax.set_ylabel('')

# === 添加 colorbar ===
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.3, 0.02, 0.5])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Pearson Correlation", fontsize=12)

plt.suptitle("Correlation", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.93, 0.96])
plt.show()
