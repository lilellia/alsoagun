""" Make the scatterplots relating bullets fired and normalized score """

import pandas as pd
import re
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def episode_plot():
    data = pd.read_csv('data/alsoagun_episode_data_full.csv')
    data['total shots'] = data['useless'] + data['utility'] + data['useful']
    data['volume'] = [re.search('(V\d)', e).group(1) for e in data['episode']]

    ax = sns.scatterplot(x='total shots', y='normalize', data=data, hue=data.volume)
    ax.axhline(y=0.5, color='#aaaaaa', linestyle='--')
    ax.set(ylim=(0, 1))

    fig = ax.get_figure()
    fig.suptitle('Normalized score vs Total shots fired (by episode)')
    fig.savefig('graphs/episode_scatter.png')


if __name__ == '__main__':
    episode_plot()
