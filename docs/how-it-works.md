# How seleCte works

The [README](../README.md) gets you generating levels. This page is the detail behind
it: every stage of the pipeline, what each module does, and the decisions that are not
obvious from the code.

## The pipeline

```
Celeste .bin                                      you supply, from your own copy
  │  data_loader/data_loader.jl                   Julia + Maple, loadMap
  ▼
data/levels/original/<Chapter>/lvl_<id>_{fg,bg,obj}.csv + _entities.json
  │                                    │
  │  preprocessing/                    │  preprocessing/
  │    recombine_fg_entities.py        │    collect_bg_rooms.py
  ▼                                    ▼
data_pcg_ready/all_rooms/              data_pcg_ready/all_rooms_bg/
        824 rooms each, generated, not committed
  │  pcg/dpe_generation.py [--layer fg|bg]        3x3 MdMC window, mask 000011012
  ▼
probability_estimation_dicts/{,bg_}*.json         the trained model, COMMITTED
  │                                               ← start here without the game
  │  pcg/level_generation.py + celeskeleton + playability
  ▼                                               (foreground only past this point)
<level>/room_N.csv + data.json
  │  room_encoder/data_to_level.jl                Julia + Maple, encodeSide
  ▼
playable .bin
```

All paths are under `src/seleCte/`.

## The layout: `celeskeleton`

The structural stage, run before any tile is sampled. A `Celeskeleton` is a mapping of
room name to `Room`, each an axis-aligned rectangle on one integer grid.
`generate_celeskeleton` grows a level as a **tree**: each new room attaches to one
existing room, so there is exactly one route from the starting room to the ending room.
`get_traversal_path` returns it, and only those rooms are held to the playability check;
everything else is an optional branch. `show_skeleton` plots the layout with matplotlib.

## The model: `pcg`

`dpe_generation.py` slides a window over the corpus and counts which symbol follows each
neighbourhood pattern, normalized per pattern. The window is the **MdMC mask**, a
flattened string where `1` marks a neighbour and `2` the cell being predicted. The
default `000011012` is `d5`:

```
0 0 0
0 1 1     ← hence `pe_d5_full.json`
0 1 2
```

`level_generation.py` wraps sampling in the skeleton, the playability check, the
placement of the spawn and the crystal heart, and the retry loops. Every retry loop is
bounded, so a run always terminates, and it says so when what it wrote is suspect.

`rooms_generation.py` is the experiment harness. It samples from a dict with no
playability check and no retries, which makes it the tool for comparing dicts rather
than for producing levels: nothing filters what it writes, so nothing biases what you
measure.

```bash
uv run python -m seleCte.pcg.rooms_generation --experiment-type rooms --nb-rooms 5 --dict-proba-estimation pe_d5_full --output-dir my_rooms
uv run python -m seleCte.pcg.rooms_generation --experiment-type skel --nb-rooms 5 --proba 0.3 --output-dir my_rooms_skel
```

The first gives loose, unconnected rooms. The second gives a connected skeleton, filled
and carved but never checked, in about a second.

## Playability

`Room.is_playable_room` runs A\* between a room's exits, via `utils.astar`. It works over
open cells, 8-connected, and models **no gravity, no jump reach, no dash, no stamina and
no wall-grab**. So "playable" currently means *a path through empty space exists*, which
is nearly always true and says little about Celeste. It certifies rooms that cannot be
played, and rejects rooms that are fine.

It is also the bottleneck, and scales badly with room area: a 7-room level of 22x14 rooms
takes a few seconds, the same level at the default 40x23 takes minutes. Most of that is
A\* rescanning its open and closed lists rather than the search itself.

Replacing it with a flood-based, directed reachability model is the main line of active
work. `astar` stays as the comparison baseline. The difficulty and interestingness
metrics live alongside it in `utils/utils.py`, and score a room along the path rather
than over the whole room.

## The Julia ends

`data_loader/data_loader.jl` decodes `.bin` maps into a per-room csv/json dump, and
`room_encoder/data_to_level.jl` encodes a generated level back into a `.bin`. Both run
from the repository root with `--project=.`, take positional arguments, and print usage
on `-h`.

`room_encoder/fixtures/round_trip_room/` is a hand-written 40x23 room you can encode
straight away, to check that your Julia setup works:

```bash
julia --project=. src/seleCte/room_encoder/data_to_level.jl src/seleCte/room_encoder/fixtures/round_trip_room round_trip_room
```

**Maple is pinned to [a fork](https://github.com/robilouis/Maple), and that is
load-bearing.** Upstream cannot encode the crystal heart, and a generated level ends on
one, so building against upstream silently produces levels that cannot be finished. The
fork also sets `HeartIsEnd`, which is what makes collecting the heart end the chapter.
To move the pin, change the single `[sources]` entry in `Project.toml`, re-instantiate,
and commit the resulting `Manifest.toml`.

## Retraining

The repository ships no Celeste level data: it is not ours to redistribute. What is
committed is `probability_estimation_dicts/`, the trained model, which is tile-frequency
statistics rather than level content.

Extract the vanilla chapters from your own copy of the game:

```bash
julia --project=. src/seleCte/data_loader/data_loader.jl "<Celeste>/Content/Maps" src/seleCte/data/levels/original
```

Each map becomes one folder holding `lvl_<id>_{fg,bg,obj}.csv` and
`lvl_<id>_entities.json` per room. Anything that is not a `.bin` is skipped. Mod dumps
such as Strawberry Jam work the same way.

Then build the corpus and estimate a new dict:

```bash
uv run python -m seleCte.pcg.preprocessing.recombine_fg_entities
uv run python -m seleCte.pcg.dpe_generation --dataset 0123456789L
```

`recombine_fg_entities` folds each room's foreground tiles and entities into the
single-symbol-per-tile grid the model trains on. Every solid tile variant collapses to
`1`, because subtypes are decoration and would have the model learn tileset transitions
instead of level shape. `--dry-run` reports what would change without writing, and
`--levels-dir` points it at a mod dump instead of the vanilla tree.

### The background layer

The same model runs over the background layer, selected with `--layer bg`:

```bash
uv run python -m seleCte.pcg.preprocessing.collect_bg_rooms
uv run python -m seleCte.pcg.dpe_generation --layer bg --dataset 6
uv run python -m seleCte.pcg.rooms_generation --layer bg -et rooms -nr 5 -dpe bg_6_000011012 -od my_bg_rooms
```

`collect_bg_rooms` only validates and renames: the game's background files are already
one symbol per cell, and nothing is collapsed, because on a decorative layer the tileset
variants *are* the signal.

**Train one chapter at a time.** A background tileset belongs to the chapter, not the
room: 674 of the 824 vanilla rooms hold exactly void plus one tileset, fixed per chapter.
Trained over all chapters at once, the model mixes seven tilesets that never appear
together in the game.

The background layer stops at an array for now: it has no skeleton integration, and the
encoder does not write backgrounds into the `.bin` yet.

## The tile alphabet

One character per tile, throughout:

```
0 void   1 solid   ^ < v > spikes   S spinner   D dreamBlock   C crumbleBlock
_ jumpThru   O greenBooster   Q redBooster   W water   B bigSpinner   R refill
F infiniteStar   L lightbeam
```

Keys are Celeste's own internal names, so two do not read as what they are: `bigSpinner`
is the bumper and `infiniteStar` is the feather. `P` (player) and `H` (crystal heart)
are placed after generation and are deliberately outside the sampled vocabulary, which
lives in `src/seleCte/utils/data/tiles_info.json`.

[ALPHABET.md](../src/seleCte/room_encoder/ALPHABET.md) documents how this alphabet maps
onto Celeste's own tiles and entities, which symbols upstream Maple cannot decode, and
where the round trip is not exact. **Read it before touching either Julia script or the
symbol vocabulary.**

## Code conventions

Ruff at line length 88 with Google-style docstrings, mypy `strict` as the gate,
[ty](https://github.com/astral-sh/ty) as the fast local loop, `pathlib` over `os.path`,
and a module-level `logger` rather than `print`.

## Where this is going

Beyond the [issue tracker](https://github.com/robilouis/seleCte/issues):

- **Replace the playability model.** Reachability under gravity is *directed*: falling
  is free, climbing is not, so per-room "is it playable" is not even well-formed. The
  design is an exit-to-exit reachability matrix computed by flooding from each exit, over
  a movement model with jump, dash, stamina and wall-grab, validated against the 824
  vanilla rooms as known-playable examples.
- **An ending generator** that chooses between a finish zone and a crystal heart,
  instead of always emitting a heart.
- **Directional bias in the layout**, so a level can be told to trend in one of eight
  directions with a tunable strength.
- **Interesting collectibles**: strawberries placed so that reaching them is a
  deliberate detour.
