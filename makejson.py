import collections
import json
import pathlib
import sqlite3
import sys

import alsoagun

if __name__ == '__main__':
    """ Update the json file from the database file. """
    records = alsoagun.get_raw_data()

    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for _, vol, chp, char, score, desc, count, isexcl, isbow in records:
        d = dict(character=char, score=score, description=desc, count=count)
        if isexcl:
            d.update(exclusive=True)
        if isbow:
            d.update(bow=True)

        data[vol][chp].append(d)

    with open(sys.argv[1], 'w+') as f:
        f.write(json.dumps(data, indent=4))
