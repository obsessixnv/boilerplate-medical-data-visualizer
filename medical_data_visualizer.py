import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
df = pd.read_csv('medical_examination.csv')
print(df.head(5))
df['overweight'] = (df['weight']/(df['height']/100)**2).apply(lambda x: 1 if x > 25 else 0)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x==1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x==1 else 1)
print(df.head(5))

# 4. Clean the data
df = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
]

# 5. Draw Categorical Plot
def draw_cat_plot():
    df_cat = pd.meld(df, id_vars = ["cardio"], value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar", height=5, aspect=1)
    fig.set_axis_labels("variable", "total")

    # 9. Save the plot
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # 11. Clean the data
    df_heat = df.copy()

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, mask=mask, square=True, fmt='.1f', center=0, vmin=-0.1, vmax=0.3, cbar_kws={"shrink": .5})

    # 16. Save the plot
    fig.savefig('heatmap.png')
    return fig
draw_heat_map()
