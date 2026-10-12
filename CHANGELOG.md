# seleCte Changelog

## [0.9.0] 04.08.2026

* Added background tile generation, as a `--layer {fg,bg}` flag through the existing
  pipeline rather than a parallel set of modules. The sweep, the estimator and the CLI
  scaffolding are shared; only the corpus, the vocabulary and the empty-room seed differ
  * `preprocessing/collect_bg_rooms.py` is the corpus stage, and is mostly the *absence*
    of the foreground one. The game's `lvl_<id>_bg.csv` files are already the
    representation the pipeline reads, so with no entities there is no pixel geometry, no
    point-entity shift and nothing to stamp; what is left is discovery, the vocabulary
    check and the `{chapter_prefix}_{filename}` rename. All 824 vanilla rooms pass, using
    exactly the fifteen symbols already declared in `valid_bg_tiles`
  * Nothing is collapsed, unlike the foreground stage. There, solid variants are
    decoration and collapsing them stops the model learning tileset transitions instead of
    level shape; on the background layer the tileset transitions are the only thing there
    is to learn
  * **Background dicts should be trained one chapter at a time.** The tileset is a
    property of the chapter, not of the room - 674 of 824 rooms hold exactly void plus one
    tileset, fixed per chapter. Estimated over all chapters at once, the dominant all-void
    pattern (1.71M of 2.58M windows) resolves to a mix of seven tilesets that never
    co-occur in the game; per chapter it resolves to the one correct tileset
* Extracted `preprocessing/corpus.py`: the csv dialect, the vocabulary check and the
  naming contract are shared by both layers, and were living in a module named for one of
  them. `invalid_symbol_counts` now takes the valid set rather than hardcoding `("0", "1")`
* `dpe_generation` gained `--layer`, and its counting loop moved into a testable
  `estimate_dpe`. Foreground dicts keep their `{dataset}_{mask}.json` name; background
  ones are prefixed `bg_`, a prefix rather than a suffix so `load_dpe` can still recover
  the mask from the tail of the file name
  * Dropped the `itertools.product` pre-seed. It allocated one empty list per *possible*
    pattern and the normalization filter dropped every unobserved one again, so it bought
    nothing - while growing as vocabulary^neighbourhood, which is what made masks wider
    than `d7` impractical: a five-cell mask would have allocated 759k lists to hold the
    ~200 patterns a corpus actually contains. Patterns are now sorted into vocabulary
    order explicitly, since without the product enumeration imposing an order the output
    would have followed `iterdir()` and been filesystem-dependent
  * `estimate_dpe` fails loudly on a missing corpus or a `--dataset` matching no room,
    instead of writing a dict with no patterns in it
  * Behaviour-preserving where it counts: retraining `--dataset 1` reproduces the
    committed `1_000011012.json` byte for byte
* Fixed `generate_new_tile`'s `all_symbols` vocabulary never being passed by any caller,
  so all three call sites in `update_room_array` fell back to `ALL_TILES_AND_ENTITIES`.
  Harmless while there was one layer; with two it means an unseen pattern in a background
  room draws a foreground symbol, writing a spinner into a decorative layer
* `generate_empty_room_data` takes a `border` symbol. The foreground border of solid `"1"`
  is the wall a room is enclosed by, which a decorative layer has no equivalent of, so
  background generation seeds with void - matching the corpus, whose outer row and column
  are 65% and 59% void
* `color_map` takes a per-layer palette. It was one hardcoded dict that raised `KeyError`
  on `2 5 7 8 9 a b c d e`, so a background room could not be displayed at all. One dict
  cannot serve both layers: `"1"` is solid ground in a recombined foreground room and a
  tileset in a background one
* Documented in `generate_room_data` that the target cell trails the window, so the first
  two rows and columns are never written. On a foreground room that reads as the wall it
  is; on a background room it is a two-cell void margin. Left as is - correcting the sweep
  bounds would change foreground output too, and invalidate the committed dicts
* Known v1 characteristic: background output under-fills. Sampling chapter 6 gives 15-45%
  tileset coverage against the chapter's real 55% mean. The border seed barely moves it
  (an `"a"` border gives 23-45%), so it belongs to the model rather than to the seed
* Scope: this generates a `height x width` array of symbols and stops there. There is no
  skeleton integration, so `rooms_generation --layer bg` rejects `-et skel`, and the Julia
  encoder does not write `lvl_<id>_bg.csv` back out yet

## [0.8.0] 31.07.2026

* Graduated `data_loader` from a notebook to `data_loader/data_loader.jl`, a CLI taking a
  maps directory and an output directory. It no longer writes the dump back into the
  folder it read the `.bin` files from, and it skips non-`.bin` files instead of relying
  on an extension check that returned the wrong thing for `a.b.bin`
* Pinned `Maple` in a root `Project.toml`, by commit rather than by branch. This replaces
  the `Pkg.add(PackageSpec(url=...))` that ran on every invocation of the encoder and
  re-resolved against whatever upstream master happened to be that day. Swapping upstream
  for a fork is now a one-line change to `[sources]`, since the UUID is the same either
  way. `JSON` is held at 0.21: `Maple` declares no compat of its own, so nothing otherwise
  stopped Pkg from resolving the 1.x rewrite against it
* Fixed `data_to_level.jl` never importing `Maple` - it called `FgTiles`, `Room`, `Map`,
  `entityMap` and `encodeMap` with no `using` for any of them, so the script could not run
  * Dropped `CSV` and `DataFrames`, which existed to read a space-separated grid of single
    characters that `readdlm` reads directly
  * The output directory is created rather than assumed, missing rooms and missing
    `data.json` entries are reported by name instead of failing inside `get(nothing, ...)`,
    and `-h` prints usage
* Documented the recombined alphabet in `room_encoder/ALPHABET.md`: how `FgTiles` and
  `entityMap` split one grid between them, the symbol-by-symbol correspondence with
  `Maple`'s `default_entity_map`, and the two symbols this project emits that upstream
  cannot decode - `L` (lightbeam), and `H` (crystal heart), whose entry exists only in a
  patched `Maple`. Also recorded where the round trip is not exact: decoding does not
  apply `Maple`'s `entity_offsets`, so point entities come back a tile up-left, and `^`
  and `<` land on the solid they stand against rather than on the cell they occupy
* Added `room_encoder/fixtures/round_trip_room`, a hand-written 40x23 room in the
  recombined alphabet with its `data.json`, encodable as-is. Harvested from the deleted
  notebook, whose saved output pins the eleven entities it must produce
* Removed `data_loader.ipynb` and `data_to_level.ipynb`

## [0.7.0] 30.07.2026

* Graduated `recombine_fg_entities` from a notebook to a module - first step towards a
  data-free repo, where the game data is extracted by whoever owns the game rather than
  shipped
  * The notebook could not run at all: its room index (`all_rooms.csv`) had been deleted,
    and its json-dump cell pointed at a directory that does not exist. Rooms are now
    discovered by globbing the levels tree
  * Fixed the entity stamping geometry: point entities landing on a room's first row or
    column were silently dropped. This recovers 903 stamps and changes 579 tiles across
    112 of the 824 rooms, all of them spinners on row 0 or column 0
  * Fixed `filter_entities` ignoring its own entity-map argument, the invalid-tile warning
    reporting the count of *valid* tiles, and logging going to the root logger
  * Dropped the notebook's copies of `color_map`, `visualize_map`, `submatrix_to_count`
    and `get_all_submatrices` - `seleCte.utils` already owns all four
  * The entity to symbol map moved into `tiles_info.json` as `entity_symbols`, so it
    cannot drift from `all_tiles_and_entities`
* Fixed a circular import between `seleCte.utils` and `seleCte.celeskeleton` that made
  `import seleCte.utils` - and therefore every PCG entry point - fail outright
* Fixed `dpe_generation` counting from an empty dict, which raised `KeyError` on its first
  iteration and made DPE regeneration impossible
* Regenerated the 10 probability estimation dicts. `pe_d5_full.json` had been trained on a
  partial corpus (161 patterns); it now covers all 824 rooms (301 patterns) and carries its
  MdMC mask, so `check_dpe_matches_mask` can verify it
* The recombined corpus is no longer committed - rebuild it with
  `python -m seleCte.pcg.preprocessing.recombine_fg_entities`. Removed the unreferenced
  `rooms_40x23/` and `40x23_fg/` artifacts

## [0.6.0] 24.07.2026

* Refactoring after repo audit to match publishability standards
  * Address broken imports
  * Fix paths
  * Migrate project to uv
  * Quality and maintainability: tests, linting, formatting...
  * Rewrote `Celeskeleton.show_skeleton` with matplotlib and dropped the `bokeh` dependency

## [0.5.1] 02.01.2026

* Small refactoring to prepare for larger improvements
  * Moved global variables to config files
  * Small changes to improve readability and maintainability in `celeskeleton`

## [0.5.0] 10.12.2024

* Refactor the whole repo to make it public - remove tests, local exp., etc.

## [0.4.0] 05.08.2024

* Implementation of all evaluation functions and helpers in `utils.py`

## [0.3.0] 06.04.2024

* Implementation of A-star like algo and integration in the PCG generation for the sake of playability
* Fusion of both `celeste_pcg_utils.py` and `celeste_playability_utils.py` into a single `utils.py` file
* Pipeline to create a level based on CLI: from parameters to binary file

## [0.2.0] 02.02.2024

* Adapted the pcg model to handle the automated creation of borders and exits for each room
* Massive reorganization of the code
* Building pipeline part between celeskeleton and room encoder

## [0.1.0] 22.12.2023

* Extended notebook in pcg model to combine celeskeleton + room generation
* Map encoding function made in Julia - next step is the exits handling

## [0.0.8] 10.10.2023

* Basic draft of AI MdMC model
* Made visual interface v1 using pandas styler
* Recombination of fg + entities in a single structure to make it possible for the model to interpret it as a single object + creation of those maps

## [0.0.7] 22.09.2023

* Worked on room size distribution
  * Generated csv files with number of rooms per unique size, room sizes and entities data
  * Generated a few plots
  * Worked a lot in `celeste_data_exploration.ipynb` - needs a function for visual interface; Ahorn-like

## [0.0.6] 19.09.2023

* Updated organization of code for skeleton
  * Added a few functions to deal with interface skeleton/JSON
  * Minor fixes to `Room` and `Celeskeleton` classes
* Slowly adapt from notebooks to Julia scripts

# [0.0.5] 07.03.2023

* Created data exploration functions
* Added first draft of playability exploration

## [0.0.4] 11.02.2023

* Finished skeleton v0

## [0.0.3] 07.02.2023

* Created the `Celeskeleton` class for the skeleton part of the project

## [0.0.2] 04.02.2023

* Created the `Room` class for the skeleton part of the project

## [0.0.1] 03.02.2023

* Created the project structure
* Made bokeh visual interface for lvl skeleton
* Finished `data_loader` to build rooms database
