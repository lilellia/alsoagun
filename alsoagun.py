import argparse
import collections
import colorsys
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from pprint import pprint
import re

# a namedtuple designed for storing the point values for each category of bullet usage
Weights = collections.namedtuple('Weights', ['useful', 'utility', 'useless'])

def calculate_score(data: tuple, weights: tuple) -> float:
    """ Calculate the gun effectiveness score based on the number of useful, utility, and useless instances.
    :param `data`: 3-tuple containing the total count of useful, utility, and useless shots (in that order)
    :param `weights`: 3-tuple containing the weights (scores) for each category. Likely a Weights(namedtuple) instance.

    :return: float; the total score for this data
    """
    return sum(w * x for w, x in zip(weights, data))

def dataframe_total_instances(dataframe: pd.DataFrame) -> int:
    """ Return the total numbers of shots in the dataframe """
    return dataframe['useful'] + dataframe['utility'] + dataframe['useless']

def normalize(data: tuple, weights: tuple) -> float:
    """ Normalize a single data point into [0, 1] based on the total number of instances. """
    total = sum(data)
    score = calculate_score(data, weights)

    m, M = min(weights), max(weights)

    n = score / total if total else (m + M) / 2
    return (n - m) / (M - m)

def dataframe_normalize(dataframe, weights):
    """ Normalize the 'score' column into [0, 1] by the total number of instances """
    m, M = min(weights), max(weights)

    dataframe['normalize'] = dataframe['score'] / dataframe_total_instances(dataframe) if dataframe_total_instances else (m + M) / 2
    dataframe['normalize'] = (dataframe['normalize'] - m) / (M - m)       # shift from [m, M] to [0, 1]

def hexcolor(r: float, g: float, b: float) -> str:
    """ Convert an rgb color to its hex string """
    R = int(r * 255)
    G = int(g * 255)
    B = int(b * 255)
    return f'#{R:02X}{G:02X}{B:02X}'

def map_colors(normalized_data):
    """ """
    def lerp(x, lower, upper, new_lower, new_upper):
        """ linearly interpolate the value `x` into [new_lower, new_upper] based on its position in [lower, upper] """
        return lower + (x - lower) / (upper - lower) * (new_upper - new_lower)

    data = normalized_data.tolist()
    res = []
    for x in data:
        h = lerp(x, -1.0, 1.0, 0, 0.6)
        r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.5)
        res.append(hexcolor(r, g, b))
    return res


def get_character_data(volumes: list, exclusive: str, ignore: list, only: list, combine_yang: bool, ignore_trailers: bool, bow: bool, compress: bool, weights: tuple):
    r""" Parse rwby_data.json for the necessary data.
    :param `volumes`: list[str] determining which volumes' data to read
    :param `exclusive`: str ("exclusive" or "mixed" or "all"), determining which weapons to read
    :param `ignore`: list[str] determining which characters to ignore
    :param `only`: list[str] determining which characters to include
    :param `combine_yang`: bool, determining whether "Yang" and "Yang*" should be combined
    :param `ignore_trailers`: bool, determining whether trailers should be ignored
        An episode is considered a trailer if its name does not match the r"Chapter \d+" regex format.
    :param `bow`: bool, determining whether bow data should be included
    :param `compress`: bool, determining whether each instance should be counted as a single instance, rather than per bullet
    :param `weights`: tuple, determining the score weights
    """
    ignore = [name.lower() for name in ignore]
    only = [name.lower() for name in only]

    with open('itsalsoagun.json', 'r') as f:
        raw = json.loads(f.read())

    # data_by_char of the form {
    #   "Ruby": {
    #       "useful": 2,
    #       "utility": 3,
    #       "useless": 2
    #   },
    #   "Weiss": ...
    # }
    data_by_char = collections.defaultdict(lambda: {'useful': 0, 'utility': 0, 'useless': 0})

    # populate data_by_char from the .json file
    for volume, volume_data in raw.items():
        v = re.search(r'Volume (\d+)', volume).group(1)
        if v not in map(str, volumes):
            continue
        for chapter, chapter_data in volume_data.items():
            if ignore_trailers and re.match(r'Chapter \d+', chapter) is None:
                continue
            for instance in chapter_data:
                c = instance['character']
                s = instance['score']
                e = instance.get('exclusive', False)

                if (excl == 'mixed' and e) or (excl == 'exclusive' and not e):
                    # ignore this data point if exclusivity query doesn't match
                    continue

                if instance.get('bow') and not bow:
                    # ignore this data point if this is a bow attack and we don't care about those
                    continue

                if c == 'Yang*' and combine_yang:
                    c = 'Yang'

                if only and c.lower() not in only:
                    continue

                if c.lower() in ignore:
                    continue

                if instance.get('count'):
                    data_by_char[c][s] += 1 if compress else instance['count']

    # convert to dataframe
    df = pd.DataFrame(data_by_char).transpose().replace(np.nan, 0.0)

    # calculate scores
    df['score'] = calculate_score((df['useful'], df['utility'], df['useless']), weights)

    # normalize (total score / instances)
    dataframe_normalize(df, weights)

    # ensure that the columns are in the right order
    cols = ['useless', 'utility', 'useful', 'score', 'normalize']
    df = df[cols]

    # make instance counts integers, normalized scores to 3d.p.
    df[cols[:3]] = df[cols[:3]].astype(int)
    df[cols[-1]] = round(df[cols[-1]], 2)

    # if "Ren" ['Lie Ren'---Team JNPR] and "Li Ren" [his father] are both in the data set,
    # replace "Ren" with "Lie Ren"
    if {'Ren', 'Li Ren'}.issubset(set(df.index.tolist())):
        df = df.replace('Ren', 'Lie Ren')

    return df


def get_character_agnostic_data(volumes: list, exclusive: str, ignore: list, only: list, ignore_trailers: bool, bow: bool, compress: bool, weights: tuple):
    """ Get episode data. Parameters have the same meaning as in get_character_data(*) """
    ignore = [name.lower() for name in ignore]
    only = [name.lower() for name in only]

    with open('itsalsoagun.json', 'r') as f:
        raw = json.loads(f.read())

    res = collections.defaultdict(list)
    for volume, volume_data in raw.items():
        v = re.search('Volume (.+)', volume).group(1)
        if v not in map(str, volumes):
            continue
        for chapter, chapter_data in volume_data.items():
            if 'Chapter' in chapter:
                c = re.search('Chapter (.+)', chapter).group(1)
                label = f'V{v}C{c}'
            else:
                if ignore_trailers:
                    continue
                c = '?'
                label = chapter

            instances = []
            for i in chapter_data:
                if only and i['character'].lower() not in only:
                    # ignore this data point if it's not a character we're isolating
                    continue
                if i['character'].lower() in ignore:
                    # ignore this data point if it's a character we're ignoring
                    continue
                if (excl == 'mixed' and i.get('exclusive')) or (excl == 'exclusive' and not i.get('exclusive')):
                    # ignore this data point if it's a weapon type (gun-exclusive or not) that we don't care about
                    continue
                if i.get('bow') and not bow:
                    # ignore this data point if it's a bow attack and we're ignoring those
                    continue
                instances.append(i)

            d = {'useful': 0, 'utility': 0, 'useless': 0}
            for inst in instances:
                count = 1 if compress else inst.get('count', 0)
                d[inst['score']] += count

            useful = d['useful']
            utility = d['utility']
            useless = d['useless']
            score = calculate_score((useful, utility, useless), weights)
            normalized = normalize((useful, utility, useless), weights)

            res['volume'].append(v)
            res['chapter'].append(c)
            res['useful'].append(useful)
            res['utility'].append(utility)
            res['useless'].append(useless)
            res['score'].append(score)
            res['normalize'].append(normalized)
            res['label'].append(label)

    # convert to dataframe
    df = pd.DataFrame(res)

    # round normalize values
    df['normalize'] = round(df['normalize'], 3)

    # reorder columns
    cols = ['label', 'useless', 'utility', 'useful', 'score', 'normalize']
    return df[cols]

def graph(data, labels, weights, yscale, savefile=None):
    n = len(data)
    indices = np.arange(n)

    # subplot layout
    fig = mpl.pyplot.figure(figsize=(16.0, 14.0))
    gs = mpl.gridspec.GridSpec(2, 5, figure=fig)
    gs.update(hspace=0.5, wspace=0)

    # Instance Count plot
    ax1 = plt.subplot(gs[0, :-1])
    ax1.set_yscale(yscale)

    plt.xticks(indices, labels)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(8)

    # adjusted useful shots
    ax1.scatter(indices, weights.useful * data.useful, color='#00a000', edgecolor='k', marker=7, s=100, label=f'Adj. Useful ({weights.useful:+})')
    ax1.axhline(y=sum(weights.useful * x for x in data.useful)/n, color='#00a000', linestyle='--')

    # adjusted utility shots
    ax1.scatter(indices, weights.utility * data.utility, color='#ffa000', edgecolor='k', marker=5, s=100, label=f'Adj. Utility ({weights.utility:+})')
    ax1.axhline(y=sum(weights.utility * x for x in data.utility)/n, color='#ffa000', linestyle='--')

    # adjusted useless shots
    ax1.scatter(indices, weights.useless * data.useless, color='#ff0000', edgecolor='k', marker=6, s=100, label=f'Adj. Useless ({weights.useless:+})')
    ax1.axhline(y=sum(weights.useless * x for x in data.useless)/n, color='#ff0000', linestyle='--')

    ax1.set_title('Instances of Gun/Bullet Usefulness')
    ax1.axhline(y=0, color='k')
    ax1.grid(True, which='both')
    ax1.legend()
    ax1.set_ylabel('# instances')

    # Score plot
    ax2 = plt.subplot(gs[1, :])

    # < a trick to handle coloring bars >
    colors = cm.Spectral(data.normalize)    # pylint: disable=E1101
                                            # ^("Module matplotlib.cm has no 'Spectral' member" -- It does, actually)
    plot = ax2.scatter(data.normalize, data.normalize, c=data.normalize, cmap='Spectral')
    plt.colorbar(plot)
    plt.cla()

    plt.xticks(indices, labels)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_rotation('vertical')
        tick.label.set_fontsize(8)

    ax2.set_title('Gun/Bullet Usefulness Score\nBars colored by normalized value (see colorbar): 0.0 = "useless"; 1.0 = "useful"')
    ax2.axhline(y=0, color='k')
    ax2.grid(True, which='both')
    ax2.set_ylabel('Score')
    ax2.set_yscale(yscale)

    ax2.bar(indices, data.score, color=colors, edgecolor='k')           # Score plot

    mean = sum(data.score) / len(data)
    ax2.axhline(y=mean, color='k', linestyle='--')

    # Circle Graph
    ax3 = plt.subplot(gs[0, -1])

    x = np.array([sum(data.useful), sum(data.utility), sum(data.useless)])
    labels = np.array([f'{t/sum(x):.2%}' for t in x] if sum(x) else [0 for _ in x])
    clr = np.array(['#00a000', '#ffa000', '#ff0000'])
    ax3.pie(x, autopct='%.2f%%', colors=clr)

    # total and normalized scores
    m, M = min(weights), max(weights)
    total = sum(data.score)
    norm = total / sum(x) if sum(x) else (m + M) / 2
    ax3.text(1, 0, f'Total: {total}\nNormalized: {(norm-m)/(M-m):.2f}', horizontalalignment='right', verticalalignment='bottom', transform=ax3.transAxes)

    if savefile:
        plt.savefig(savefile, dpi=200)
    else:
        plt.show()


def get_unfinished_data(volumes):
    with open('itsalsoagun.json', 'r') as f:
        data = json.loads(f.read())

    found = 0
    episodes = []
    for vol, volume_data in data.items():
        volnum = re.search(r'\d+', vol).group()
        if volnum not in volumes:
            continue
        for chp, chapter_data in volume_data.items():
            c = re.search('(Chapter )*(.*)', chp).group(2)
            unfinished = [inst for inst in chapter_data if not inst.get('count')]
            if unfinished:
                episodes.append(f'V{volnum}C{c}')
                print(f'\n{vol} {chp}')
                pprint(unfinished)
                found += len(unfinished)

    print(f'\n{found} unfinished instance(s) found.')
    print(f'Episodes: {episodes}')


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--volumes', nargs='+', default=['1', '2', '3', '4', '5', '6'], help="The volumes to consider. Trailers are associated with the following volume (so \"Red\" is part of V1, Ruby Character Short is part of V4, Weiss/Blake/Yang Character Shorts are part of V5, and Adam Character Short is part of V6. [DEFAULT: all volumes]")
    parser.add_argument('--verbose', action='store_true', help="Determines whether the data table should also be printed. [DEFAULT: False]")
    parser.add_argument('--nograph', dest='no_graph', action='store_true', help="Determine whether the graph should be omitted. Possibly useful in conjunction with --verbose. [DEFAULT: False]")
    parser.add_argument('-c', '-char', '--character', dest='character', action='store_true')
    parser.add_argument('-i', '-ignore', dest='ignore', nargs='+', default=[], help="The data associated with the people following this flag are ignored. For example, '-ignore ruby' will pull everybody's data EXCEPT for Ruby's.")
    parser.add_argument('-o', '-only', dest='only', nargs='+', default=[], help="Only the data associated with the people following this flag are considered. For example, '-only ruby weiss blake yang --combine-yang' will only show the data associated with Team RWBY.")
    parser.add_argument('-cy', '--combine-yang', dest='combine_yang', action='store_true', help="Yang's data is separated based on whether she has activated her Semblance: 'Yang' (without Semblance) and 'Yang*' (with). This flag combines these two into one character. Only useful with the --character flag.")
    parser.add_argument('-b', '--bow', dest='bow', action='store_true', help="Some characters (like Cinder and Li Ren) use bows as a weapon. This flag includes the data where they use their bow. [DEFAULT: False (do not include)] Note: Nebula's crossbow is counted regardless. Other bows (Cinder, Li Ren) are not counted if -m is passed. Other projectiles, such as Cinder's Dust attacks, Amber's rain of ice leaves, or Weiss's ice crystals are not counted.")
    parser.add_argument('--ignore-trailers', dest='ignore_trailers', action='store_true', help="The trailers often show the characters at a level we don't normally see them in the rest of the show. Passing this flag ensures that these episodes are skipped: \"Red\", \"White\", \"Black\", \"Yellow\", Ruby's CS (V4), Weiss's/Blake's/Yang's CS (V5), Adam's CS (V6). [DEFAULT: False]")
    parser.add_argument('--compress', dest='compress', action='store_true', help="Rather than counting every bullet fired, count only the instances (in the data .json file, behave as if count==1 for all instances). [DEFAULT: False]")
    parser.add_argument('-w', '--weights', dest='weights', nargs=3, type=float, default=[1.0, 0.5, -1.0], help="Set the weights of each category. They should be passed in as '-w USEFUL UTILITY USELESS'. [DEFAULT: 1.0 0.5 -1.0]")
    parser.add_argument('-s', '--scale', dest='yscale', default='symlog', help="Sets the y-axis scaling for the graph. Possible options include 'linear', 'log', 'logit' (logistic), 'symlog' (symmetric log). [DEFAULT: symlog]")
    parser.add_argument('--output', '--csv', dest='output', help="Output the data to a .csv file. Ignored if --verbose is not passed.")
    parser.add_argument('--save', dest='savefile')

    exclusivity = parser.add_mutually_exclusive_group()
    exclusivity.add_argument('-a', '--all', action='store_true', default=True, help='Include all weapons (subject to -only and -ignore flags) [DEFAULT: True] Only one of -a, -m, and -e can be passed.')
    exclusivity.add_argument('-m', '--mixed', action='store_true', help="In addition to -only and -ignore flags, consider only those weapons which are not exclusively guns (e.g. Coco's minigun, but not Junior's men's machine guns). [DEFAULT: show all weapons] Only one of -a, -m, and -e can be passed.")
    exclusivity.add_argument('-e', '--exclusive', action='store_true', help="In addition to -only and -ignore flags, consider only those weapons which are exclusively guns. [DEFAULT: show all weapons] Only one of -a, -m, and -e can be passed. Bows are not counted unless -b is also passed.")

    parser.add_argument('--todo', action='store_true')

    args = parser.parse_args()

    if args.todo:
        get_unfinished_data(volumes=args.volumes)
    else:
        if args.mixed:
            excl = 'mixed'
        elif args.exclusive:
            excl = 'exclusive'
        else:
            excl = 'all'

        weights = Weights(*args.weights)
        kwargs = {
            'volumes': args.volumes,
            'exclusive': excl,
            'ignore': args.ignore,
            'only': args.only,
            'ignore_trailers': args.ignore_trailers,
            'bow': args.bow,
            'compress': args.compress,
            'weights': weights,
        }

        if args.character:
            # graph character data
            kwargs.update(combine_yang=args.combine_yang)
            df = get_character_data(**kwargs).sort_values(by=['score'])
            labels = df.index
        else:
            # graph episode data
            df = get_character_agnostic_data(**kwargs)
            labels = df.label

        if args.verbose:
            print_full_dataframe(df)
            print()

            if args.output:
                df.to_csv(args.output)

        norm = normalize(
            data=(sum(df['useful']), sum(df['utility']), sum(df['useless'])),
            weights=Weights(*args.weights)
        )

        r = 9 * norm + 1
        black_stars = '\u2605' * round(r/2)
        white_stars = '\u2606' * (5 - round(r/2))
        print(f'Rating: {r:.1f}/10 ({black_stars+white_stars})')

        if not args.no_graph:
            graph(df, labels, weights, args.yscale, args.savefile)
