import pathlib
import subprocess
import sys
import tqdm

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

    with tqdm.tqdm(total=len(options)+2) as t:
        # graphs
        for option in options:
            outfile = pathlib.Path() / 'graphs' / option.replace('-', '')
            args = [interpreter, 'alsoagun.py'] + option.split(' ') + ['--save', outfile]
            subprocess.run(args, capture_output=True)
            t.update(1)

        # .csv files
        data = pathlib.Path() / 'data'
        args = [interpreter, 'alsoagun.py', '--nograph', '--verbose', '--csv']

        subprocess.run(args + [data / 'alsoagun_character_data.csv', '-c', '-cy'], capture_output=True)
        t.update(1)

        subprocess.run(args + [data / 'alsoagun_episode_data.csv', '-a'], capture_output=True)
        t.update(1)
