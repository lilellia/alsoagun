# ALSO A GUN

## Most Recent Data Update: Volume 6 Chapter 5 "The Coming Storm"
Note: V6C4 and V6C5 do not include any bullets fired.

So... have you noticed that so many weapons in the RWBY universe are guns, even when that doesn't make sense, like Crescent Rose? Well, how good are they? I've counted *every* bullet from *every* episode, and put them in one of three categories:

* useful: The bullet actually is useful in killing something.
* utility: The bullet serves some utility, such as creating a distraction or providing movement options using the gun's recoil.
* useless: The bullet doesn't do anything (misses are also in this category).

Note that small non-bullet projectiles, such as Cinder's Dust fragments or Weiss's ice shards are not included. Magic (such as Amber's leaf-ice shards or the magic in the V6C3 flashback) is also not included. By default, arrows are not counted, though they are in the data (see the `-b` flag, below)

## ./data

Each instance of bullets being fired is given in `./data/alsoagun.sqlite3` and `./data/itsalsoagun.json` for your use. These data are compiled into character-wise data (`./data/alsoagun_character_data.csv`) and episode-wise data (`./data/alsoagun_episode_data.csv`). Notice that any character/episode without any data isn't given in these tables.

## ./graphs

`alsoagun.py` creates graphs showing different cases. See the command line arguments to `alsoagun.py` for references to the filenames here.

## alsoagun.py

The bulk of the project, reading and processing the data. It supports a number of querying options passed as command line arguments:

* `-v V [V ...]`, `--volumes V [V ...]` :: The volumes to consider. Trailers are associated with the following volume (so \"Red\" is part of V1, Ruby Character Short is part of V4, Weiss/Blake/Yang Character Shorts are part of V5, and Adam Character Short is part of V6. [DEFAULT: all volumes]
* `'--verbose` :: Determines whether the data table should also be printed. [DEFAULT: False]
* `--nograph` Determine whether the graph should be omitted. Possibly useful in conjunction with --verbose. [DEFAULT: False]
* `-c`, `-char`, `--character` :: Process character data [DEFAULT: episode data]
* `-i CHAR [CHAR ...]`, `-ignore CHAR [CHAR ...]` :: The data associated with the people following this flag are ignored. For example, `-ignore ruby` will pull everybody's data EXCEPT for Ruby's. [DEFAULT: no characters ignored]
* `-o CHAR [CHAR ...]`, `-only CHAR1 [CHAR ...]` :: Only the data associated with the people following this flag are considered. For example, `-only ruby weiss blake yang --combine-yang` will only show the data associated with Team RWBY. [DEFAULT: show all characters]
* `-cy`, `--combine-yang` :: Yang's data is separated based on whether she has activated her Semblance: 'Yang' (without Semblance) and 'Yang*' (with). This flag combines these two into one character. Only useful with the `--character` flag. [DEFAULT: separate Yang and Yang*]
* `-b`, `--bow` ;; Some characters (like Cinder and Li Ren) use bows as a weapon. This flag includes the data where they use their bow. [DEFAULT: False (do not include)] Note: Nebula's crossbow is counted regardless. Other bows (Cinder, Li Ren) are not counted if `-m` is passed. Other projectiles, such as Cinder's Dust attacks, Amber's rain of ice leaves, or Weiss's ice crystals are not counted.
* `--ignore-trailers` :: The trailers often show the characters at a level we don't normally see them in the rest of the show. Passing this flag ensures that these episodes are skipped: \"Red\", \"White\", \"Black\", \"Yellow\", Ruby's CS (V4), Weiss's/Blake's/Yang's CS (V5), Adam's CS (V6), and all volume intros. [DEFAULT: False (i.e. include all episodes)]
* `--compress` :: Rather than counting every bullet fired, count only the instances [DEFAULT: False (i.e. count each bullet)]
* `-w USEFUL UTILITY USELESS`, `--weights USEFUL UTILITY USELESS` :: Set the weights of each category. [DEFAULT: 1.0 0.5 -1.0]
* `-s SCALE`, `--scale SCALE` :: Sets the y-axis scaling for the graph. Possible options include 'linear', 'log', 'logit' (logistic), 'symlog' (symmetric log). [DEFAULT: symlog]
* `--output FILE`, `--csv FILE` :: Output the data to a .csv file. Ignored if `--verbose` is not passed.
* `--save FILE` :: Save the graph to the given file.

Only one of the following may be passed:
* `-a`, `--all` :: Include all weapons (subject to `-only` and `-ignore` flags) [DEFAULT: True]
* `-m`, `--mixed` :: In addition to `-only` and `-ignore` flags, consider only those weapons which are not exclusively guns (e.g. Coco's minigun, but not Junior's men's machine guns). [DEFAULT: show all weapons]
* `-e`, `--exclusive` :: In addition to `-only` and `-ignore` flags, consider only those weapons which are exclusively guns. [DEFAULT: show all weapons]

## makejson.py

This script is run to take the database file and automagically update the .json file. This allows consistent data being stored in both formats without having to manually place it twice.

It should be called as:
`$ python3 makejson.py JSON_OUTFILE`

## update.py

This script updates everything:

* Generates new graphs (calls `python3 alsoagun.py` with several combinations of flags. See `./graphs/` to see these flags.)
* Updates the CSV files (calls `python alsoagun.py --csv ./data/alsoagun_character_data.csv` and `python alsoagun.py --csv ./data/alsoagun_epsiode_data.csv`)
* Updates the JSON file according to the database (calls `python3 makejson.py data/itsalsoagun.json`)
