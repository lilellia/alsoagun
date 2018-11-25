import pandas as pd
import pathlib
import subprocess
import sys
import tqdm

import scatter


def add_blank_episodes():
    """ Add all of the blank episodes to the .csv, i.e., those without bullets fired
    This is necessary since they aren't in the database (because there aren't any bullets)
    to be recorded. This also only affects the episode data .csv file; blank characters are
    not adjusted.
    """
    partial_data = pd.read_csv('data/alsoagun_episode_data.csv').drop(columns=['Unnamed: 0'])

    # build the framework for the big dataframe
    with open('data/episode_list.txt') as f:
        episodes = [line.strip() for line in f]
    full_data = pd.DataFrame(episodes, columns=['episode'])
    for col in ('useless', 'utility', 'useful', 'score'):
        full_data[col] = [0] * len(full_data)
    full_data['normalize'] = [0.5] * len(full_data)

    # populate the big dataframe
    full_data.set_index('episode', inplace=True)
    full_data.update(partial_data.set_index('episode'))
    full_data.reset_index()

    # fix the rounding
    full_data['normalize'] = round(full_data['normalize'], 3)

    # output
    full_data.to_csv('data/alsoagun_episode_data_full.csv')


if __name__ == '__main__':
    options = [
        '-c -cy -a',        # character data, all
        '-c -cy -e',        # character data, gun-exclusive only
        '-c -cy -m',        # character data, mixed-weapons only,
        '-c -cy -only yang',# character data, Yang only
        '-a',               # episode data, all
        '-e',               # episode data, gun-exclusive only
        '-m',               # episode data, mixed-weapons only
    ]

    interpreter = pathlib.Path(sys.executable).name

    with tqdm.tqdm(total=len(options)+5) as t:
        # .csv files
        t.set_description('Updating .csv files...')
        data = pathlib.Path() / 'data'
        args = [interpreter, 'alsoagun.py', '--nograph', '--verbose', '--csv']

        subprocess.run(args + [data / 'alsoagun_character_data.csv', '-c', '-cy'], capture_output=True)
        t.update(1)

        subprocess.run(args + [data / 'alsoagun_episode_data.csv', '-a'], capture_output=True)
        t.update(1)

        add_blank_episodes()
        t.update(1)

        # json file
        t.set_description('Updating .json file...')
        args = [interpreter, 'makejson.py', pathlib.Path() / 'data' / 'itsalsoagun.json']
        subprocess.run(args, capture_output=True)
        t.update(1)

        # graphs
        t.set_description('Making graphs...')
        for option in options:
            outfile = pathlib.Path() / 'graphs' / option.replace('-', '')
            args = [interpreter, 'alsoagun.py'] + option.split(' ') + ['--save', outfile]
            subprocess.run(args, capture_output=True)
            t.update(1)

        scatter.episode_plot()
        t.update(1)
