# seleCte

Procedurally generated [Celeste](https://www.celestegame.com/) levels that the game can
load and you can play.

seleCte learns the tile patterns of Celeste's 824 vanilla rooms with a multi-dimensional
Markov chain, draws a level layout, fills each room by sampling from the model, and
encodes the result as a `.bin` map. It started as a Master's thesis and is an active
research project: the levels load, start on a player spawn and end on a crystal heart,
but they are not yet reliably *good*. See [Limitations](#limitations).

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which also installs the right Python for you.
You do **not** need a copy of Celeste: the trained model ships with the repository.

```bash
git clone https://github.com/robilouis/seleCte
cd seleCte
uv sync
uv run python -m seleCte.pcg.level_generation --nb-rooms 7 --proba 0.3 --room-size 22 14 --tries-limit 4 --level-name my_level
```

After a few seconds you have `my_level/`: one `room_<n>.csv` per room, and a `data.json`
describing the layout.

### Play it

Turning the level into a map needs [Julia](https://julialang.org/) 1.11 or later:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. src/seleCte/room_encoder/data_to_level.jl my_level my_level
```

That writes `generated_bin_files/my_level.bin`. Drop it into an
[Everest](https://everestapi.github.io/) mod folder and pick it from the chapter list.

### Options worth knowing

| Flag | What it does |
|---|---|
| `--nb-rooms` | How many rooms the level has. |
| `--proba` | Layout shape: `0` is a straight corridor, values near `1` a labyrinth. |
| `--room-size` | Width then height of each room. The default `40 23` works but takes minutes. |
| `--level-name` | Output folder. |
| `--reset-skeleton` | Redraw the whole level when a room never comes out playable. |

Every script prints its full options with `--help`.

## How it works

```
your copy of Celeste ──► extract levels ──► train the Markov model ──► model (committed)
                                                                              │
                     playable .bin ◄── encode ◄── fill rooms ◄── draw layout ◄┘
```

The right-hand half is all you need to generate levels. The left-hand half is only for
retraining, and needs the game. [docs/how-it-works.md](docs/how-it-works.md) walks
through each stage, the modules, and the tile alphabet.

## Limitations

- **"Playable" is approximate.** The check looks for a path through empty space and
  ignores gravity, jumping, dashing and stamina. It accepts rooms that cannot be
  played and rejects some that can. Replacing it is the main line of active work.
- **Large rooms are slow.** The playability check scales badly with room size.
- **Nothing that moves.** Zip movers, moving blocks, falling blocks and the like are
  dropped.
- **Partial entity coverage.** About a tenth of Celeste's entities are supported;
  lightning and springs are the notable gaps.
- **Backgrounds stop short.** Background tiles can be generated but not yet written into
  a map.

Known issues and planned work are in the
[issue tracker](https://github.com/robilouis/seleCte/issues).

## Retraining the model

This needs your own copy of Celeste. No game data is shipped here.

```bash
julia --project=. src/seleCte/data_loader/data_loader.jl "<Celeste>/Content/Maps" src/seleCte/data/levels/original
uv run python -m seleCte.pcg.preprocessing.recombine_fg_entities
uv run python -m seleCte.pcg.dpe_generation --dataset 0123456789L
```

The first command extracts every room from the game's maps, the second turns them into
training data, and the third trains a new model. Details, including the background
layer, are in [docs/how-it-works.md](docs/how-it-works.md#retraining).

## Development

```bash
git add -A
uvx pre-commit run --all-files
uv run mypy --no-pretty
uv run pytest -q
julia --project=. tests/julia_smoke.jl
```

Stage first: pre-commit only sees files git knows about. All four checks also run in CI.

## Citing

This repository is the implementation behind:

> Louis Robinet, Marcello A. Gómez-Maureira, and Mike Preuss. 2025. **Towards a Celeste
> AI Framework: Agent-free Automated 2D Level Generation for Multidirectional
> Platformers.** In *FDG '25: Proceedings of the 20th International Conference on the
> Foundations of Digital Games*. https://doi.org/10.1145/3723498.3723796

GitHub's *Cite this repository* button gives the BibTeX entry.

The paper describes an earlier version of the code, from December 2024. It has been
substantially reworked since; [CHANGELOG.md](CHANGELOG.md) lists what changed.

## License

MIT, see [LICENSE](LICENSE).

Celeste is © [Maddy Makes Games](https://www.maddymakesgames.com/) / Extremely OK Games.
This is an unaffiliated, non-commercial research project, and no game assets or level
data are redistributed here. The Julia scripts build on
[Maple](https://github.com/CelestialCartographers/Maple) by the Celestial Cartographers,
pinned to [a fork](https://github.com/robilouis/Maple) that can encode the crystal heart.
