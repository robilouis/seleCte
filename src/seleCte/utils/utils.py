"""Shared helpers for room content: MdMC sampling, A* pathfinding and metrics.

Everything here works on one representation - a room as a `(height, width)`
array of single-character tile codes. The vocabulary behind those codes (which
symbols exist, which are lethal, which are valid foreground or background
tiles) is read at import time from `data/tiles_info.json`, along with the
neighbourhood masks in `data/mdmc_matrices.json`.

- Estimation: `get_all_submatrices` and `submatrix_to_count` reduce a corpus of
  rooms to the pattern counts a probability dict is built from.
- Generation: `extract_pattern`, `generate_new_tile` and `generate_room_data`
  sample a room back out of such a dict, tile by tile.
- Pathfinding: `astar` and its `Node` helper, behind the playability checks.
- Scoring: `evaluate_room_difficulty` and `evaluate_room_interestingness` rate
  a room along a path, from entity density and tile entropy.
"""

import heapq
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from pandas.io.formats.style import Styler

if TYPE_CHECKING:
    from seleCte.celeskeleton.celeskeleton import Room

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)


DATA_DIR: Path = Path(__file__).resolve().parent / "data"

tiles_info_path: Path = DATA_DIR / "tiles_info.json"
with tiles_info_path.open(encoding="utf-8") as f:
    # `Any` rather than `dict[str, list[str]]`: `entity_symbols` is a mapping, not a
    # list, so the file is no longer homogeneous. Each entry is narrowed on binding.
    tiles_info: dict[str, Any] = json.load(fp=f)
    ALL_TILES_AND_ENTITIES: list[str] = tiles_info["all_tiles_and_entities"]
    LETHAL_ENTITIES: list[str] = tiles_info["lethal_entities"]
    NON_LETHAL_ENTITIES: list[str] = tiles_info["non_lethal_entities"]
    VALID_FG_TILES: list[str] = tiles_info["valid_fg_tiles"]
    VALID_BG_TILES: list[str] = tiles_info["valid_bg_tiles"]
    # Which game entity becomes which tile symbol during preprocessing. It lives here
    # rather than in the preprocessing module so it cannot drift from
    # `ALL_TILES_AND_ENTITIES` - a symbol missing from that list would be produced by
    # recombination and then never sampled back out by the model.
    ENTITY_SYMBOLS: dict[str, str] = tiles_info["entity_symbols"]

mdmc_matrices_path: Path = DATA_DIR / "mdmc_matrices.json"
with mdmc_matrices_path.open(encoding="utf-8") as f:
    MDMC_MATRICES: dict[str, np.ndarray[tuple[int, int], np.dtype[Any]]] = {
        name: np.array(object=matrix) for name, matrix in json.load(fp=f).items()
    }

# The neighbourhood mask every DPE dict shipped so far was trained with.
DEFAULT_MDMC_MATRIX: str = "000011012"

# Reserved top-level key holding the mask inside a DPE file. `#` is not a tile
# symbol, so this can never collide with a pattern key.
MDMC_KEY: str = "#mdmc_matrix"


# Functions for MdMC neighbourhood masks
def parse_mdmc_matrix(flat: str) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
    """Parses a flattened mask string into the square matrix it describes.

    A mask marks the cells of a sliding window with `1` for the neighbours that
    make up a pattern and `2` for the single tile that pattern predicts; every
    other cell is ignored. The default `"000011012"` is the 3x3 window::

        0 0 0
        0 1 1
        0 1 2

    Args:
        flat: the mask, row-major, one digit per cell.

    Returns:
        The mask as an int matrix.

    Raises:
        ValueError: the string is not a perfect square, has no neighbourhood
            cell, or does not have exactly one target cell.
    """
    side = round(np.sqrt(len(flat)))
    if side * side != len(flat):
        raise ValueError(
            f"MdMC mask {flat!r} has {len(flat)} cells, which is not a square."
        )
    matrix: np.ndarray[tuple[int, int], np.dtype[Any]] = np.array(
        object=[int(c) for c in flat]
    ).reshape(side, side)
    if not (matrix == 1).any():
        raise ValueError(f"MdMC mask {flat!r} has no neighbourhood cell (a `1`).")
    if int((matrix == 2).sum()) != 1:
        raise ValueError(f"MdMC mask {flat!r} needs exactly one target cell (a `2`).")

    return matrix


def mdmc_offsets(
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]],
) -> tuple[tuple[int, int], ...]:
    """Returns the neighbourhood cell offsets (the `1`s), in row-major order."""
    return tuple((int(i), int(j)) for i, j in np.argwhere(mdmc_matrix == 1))


def mdmc_target_offset(
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]],
) -> tuple[int, int]:
    """Returns the offset of the tile a pattern predicts (the single `2`)."""
    i, j = np.argwhere(mdmc_matrix == 2)[0]
    return int(i), int(j)


DEFAULT_MDMC_OFFSETS: tuple[tuple[int, int], ...] = mdmc_offsets(
    parse_mdmc_matrix(DEFAULT_MDMC_MATRIX)
)
DEFAULT_MDMC_TARGET: tuple[int, int] = mdmc_target_offset(
    parse_mdmc_matrix(DEFAULT_MDMC_MATRIX)
)


def load_dpe(
    path: str | Path,
) -> tuple[
    dict[str, dict[str, int | float]], np.ndarray[tuple[int, int], np.dtype[Any]]
]:
    """Reads a DPE file and recovers the mask it was trained with.

    Dicts written before the mask was persisted do not carry it, so it is
    recovered from the `{dataset}_{mask}.json` filename, and failing that
    assumed to be `DEFAULT_MDMC_MATRIX`.

    Args:
        path: the json file to read.

    Returns:
        The `{pattern: {symbol: probability}}` dict and its mask.
    """
    dpe_path = Path(path)
    with dpe_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(fp=f)

    flat: str | None = raw.pop(MDMC_KEY, None)
    if flat is None:
        suffix: str = dpe_path.stem.rsplit(sep="_", maxsplit=1)[-1]
        flat = suffix if suffix.isdigit() else DEFAULT_MDMC_MATRIX
        logger.warning(
            msg=f"{dpe_path.name} stores no MdMC mask - assuming {flat!r}. "
            "Re-run dpe_generation to embed it."
        )

    return raw, parse_mdmc_matrix(flat)


# Functions for DPE extraction
def submatrix_to_count(
    array: np.ndarray[tuple[int, int], np.dtype[Any]],
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]],
) -> tuple[str, str]:
    """Splits one window into the pattern it holds and the tile that follows it.

    This is the single observation a DPE is counted from: the neighbourhood
    cells of the mask become the pattern key, and the target cell is the symbol
    observed for that key.

    Args:
        array: a window of tile symbols, the same shape as the mask.
        mdmc_matrix: the neighbourhood mask, see `parse_mdmc_matrix`.

    Returns:
        The pattern and the symbol it predicted.
    """
    ti, tj = mdmc_target_offset(mdmc_matrix)
    return (
        "".join(array[i, j] for i, j in mdmc_offsets(mdmc_matrix)),
        array[ti, tj],
    )


def get_all_submatrices(
    array: np.ndarray[tuple[int, int], np.dtype[Any]], size: int = 2
) -> list[np.ndarray[tuple[int, int], np.dtype[Any]]]:
    """Slides a square window over a room and returns every position of it.

    Args:
        array: the room, as a `(height, width)` array of tile symbols.
        size: window offset, so the windows returned are `size + 1` a side.

    Returns:
        Every window, in row-major order. Windows are views into `array`.
    """
    xmax, ymax = array.shape
    all_submatrices: list[np.ndarray[tuple[int, int], np.dtype[Any]]] = []
    for x in range(xmax - size):
        for y in range(ymax - size):
            all_submatrices.append(array[x : x + size + 1, y : y + size + 1])
    return all_submatrices


# Functions for visualization
# One palette per layer, because the two alphabets overlap without agreeing: `"1"`
# is solid ground in a recombined foreground room and a tileset in a background
# one, and `"0"` is the only symbol that means the same thing in both. `"p"` and
# `"e"` are not tiles at all - `visualize_room_path` writes them over a room to
# draw an A* path and its zone of interest.
FG_PALETTE: dict[str, str | None] = {
    "1": "salmon",
    "<": "grey",
    ">": "grey",
    "v": "grey",
    "^": "grey",
    "P": "pink",
    "W": "blue",
    "D": "purple",
    "C": "grey",
    "_": "brown",
    "O": "green",
    "Q": "red",
    "S": "orange",
    "B": "cyan",
    "R": "lime",
    "F": "yellow",
    "L": "lightblue",
    "H": "black",
    "0": None,
    "p": "violet",
    "e": "plum",
}

# Background tilesets are decoration, so the colours only have to be told apart -
# there is no solid/lethal/safe distinction to encode. Void stays uncoloured, so a
# generated room reads as the shape it is.
BG_PALETTE: dict[str, str | None] = {
    "0": None,
    "1": "salmon",
    "2": "orange",
    "3": "gold",
    "4": "khaki",
    "5": "yellowgreen",
    "6": "mediumseagreen",
    "7": "turquoise",
    "8": "skyblue",
    "9": "cornflowerblue",
    "a": "mediumpurple",
    "b": "orchid",
    "c": "palevioletred",
    "d": "rosybrown",
    "e": "silver",
    "p": "violet",
}

PALETTES: dict[str, dict[str, str | None]] = {"fg": FG_PALETTE, "bg": BG_PALETTE}


def color_map(val: str, palette: dict[str, str | None] = FG_PALETTE) -> str | None:
    """Returns the css background colour a tile symbol is drawn with.

    Args:
        val: a tile symbol, as stored in a room array.
        palette: the layer's symbol-to-colour map, see `PALETTES`.

    Returns:
        The css declaration to hand to a pandas `Styler`.

    Raises:
        KeyError: the symbol is not in the palette, which means the room is being
            drawn with the other layer's colours.
    """
    return f"background-color: {palette[val]}"


def visualize_room(df: pd.DataFrame, layer: str = "fg") -> Styler:
    """Colours a room the way `color_map` describes, for notebook display.

    Args:
        df: a room, one tile symbol per cell.
        layer: which layer's palette to draw with, `"fg"` or `"bg"`.

    Returns:
        The styled frame.
    """
    palette: dict[str, str | None] = PALETTES[layer]

    # `Styler.map` is typed as taking a callable handling every scalar type,
    # since a frame can hold anything. A room only ever holds tile symbols.
    return df.style.map(
        func=lambda val: color_map(val=cast("str", val), palette=palette)
    )


def generate_empty_room_data(
    width: int = 40, height: int = 23, border: str = "1"
) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
    """Builds the empty room generation starts from: void inside a border.

    The border is what gives the first sweep positions a defined neighbourhood to
    read, so generation has something to condition on before any tile has been
    sampled. On the foreground layer it is solid `"1"`, which doubles as the wall
    a room is enclosed by. A background layer has no walls - nothing there is
    solid - so it seeds with void instead, which is also what the corpus shows:
    the outer row and column of a real background room are 65% and 59% void.

    Args:
        width: room width, in tiles.
        height: room height, in tiles.
        border: the symbol to seed the outer row and column with.

    Returns:
        A `(height, width)` array of `"0"` inside a one-tile `border`.
    """
    room_data: np.ndarray[tuple[int, int], np.dtype[Any]] = np.zeros(
        shape=(height, width), dtype=int
    ).astype(dtype=str)
    room_data[0, :] = border
    room_data[:, 0] = border
    room_data[-1, :] = border
    room_data[:, -1] = border

    return room_data


# Helper functions for extracting pattern proba for MdMC model
def extract_pattern(
    array: np.ndarray[tuple[int, int], np.dtype[Any]],
    x: int,
    y: int,
    offsets: tuple[tuple[int, int], ...] = DEFAULT_MDMC_OFFSETS,
) -> str:
    """Reads the pattern a window anchored at `(x, y)` currently holds.

    Args:
        array: the room being generated.
        x: row of the window's top-left corner.
        y: column of the window's top-left corner.
        offsets: neighbourhood offsets of the mask in use.

    Returns:
        The symbols at those offsets, joined in mask order - the key to look
        up in a DPE dict.
    """
    return "".join(array[x + i, y + j] for i, j in offsets)


def extract_proba_from_tile(
    array: np.ndarray[tuple[int, int], np.dtype[Any]],
    x: int,
    y: int,
    d_pe: dict[str, dict[str, int | float]],
    offsets: tuple[tuple[int, int], ...] = DEFAULT_MDMC_OFFSETS,
) -> dict[str, int | float]:
    """Looks up the distribution the DPE gives for the pattern at `(x, y)`.

    Args:
        array: the room being generated.
        x: row of the window's top-left corner.
        y: column of the window's top-left corner.
        d_pe: the `{pattern: {symbol: probability}}` dict.
        offsets: neighbourhood offsets of the mask in use.

    Returns:
        The distribution over next symbols, or an empty dict if the pattern
        was never seen during estimation.
    """
    pattern: str = extract_pattern(array, x, y, offsets=offsets)

    try:
        proba_dist: dict[str, int | float] = d_pe[pattern]
    except KeyError:  # unseen state
        proba_dist = {}

    return proba_dist


def generate_new_tile(
    proba_dist: dict[str, int | float],
    all_symbols: list[str] = ALL_TILES_AND_ENTITIES,
    exclude: list[str] | None = None,
) -> str:
    """Samples the next tile symbol from a pattern's distribution.

    An empty `proba_dist` means the pattern was never seen, and the symbol is
    drawn uniformly instead - which is why a DPE trained with the wrong mask
    degrades into noise rather than failing outright, see
    `check_dpe_matches_mask`.

    Args:
        proba_dist: distribution over symbols, empty for an unseen pattern.
        all_symbols: the vocabulary to fall back on for an unseen pattern.
        exclude: symbols backtracking has already tried and rejected here.

    Returns:
        The sampled symbol.

    Raises:
        AssertionError: every symbol with a non-zero probability is excluded,
            which is the signal for the caller to backtrack.
    """
    if exclude is None:
        exclude = []
    if not proba_dist:  # unseen state
        new_tile: str = np.random.choice(a=all_symbols)
    else:
        if [key for key in proba_dist if proba_dist[key] > 0] == exclude:
            raise AssertionError(
                "There is no symbol with a strictly positive proba left!"
            )
        new_tile = np.random.choice(
            a=list(proba_dist.keys()), p=list(proba_dist.values())
        )
        if new_tile in exclude:
            new_tile = np.random.choice(
                a=list(proba_dist.keys()), p=list(proba_dist.values())
            )

    return new_tile


def update_room_array(
    array: np.ndarray[tuple[int, int], np.dtype[Any]],
    x: int,
    y: int,
    d_proba_estimation: dict[str, dict[str, int | float]],
    bt_depth: int = 0,
    verbose: bool = True,
    offsets: tuple[tuple[int, int], ...] = DEFAULT_MDMC_OFFSETS,
    target: tuple[int, int] = DEFAULT_MDMC_TARGET,
    all_symbols: list[str] = ALL_TILES_AND_ENTITIES,
) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
    """Samples the single tile the window at `(x, y)` predicts, in place.

    With `bt_depth` at 0 the tile is sampled straight from the pattern's
    distribution. Above 0, the next `bt_depth` tiles to the right are sampled
    on a copy first: a choice that leaves a later position with no symbol left
    is rejected and retried, so the tile written back is one that admits a
    continuation. Both the loop-detection and the exhausted-search cases give
    up and fall back to a uniform draw.

    Args:
        array: the room being generated, modified in place.
        x: row of the window's top-left corner.
        y: column of the window's top-left corner.
        d_proba_estimation: the `{pattern: {symbol: probability}}` dict.
        bt_depth: how far ahead to look before committing to a symbol.
        verbose: log every tile written.
        offsets: neighbourhood offsets of the mask in use.
        target: offset of the tile the pattern predicts.
        all_symbols: the layer's vocabulary, drawn from uniformly whenever a
            pattern was never seen. Passing the wrong one is how a background
            room ends up holding a spinner.

    Returns:
        The same array, with the target cell filled in.
    """
    ti, tj = target
    bt_depth_true: int = min(bt_depth, array.shape[1] - y - 3)
    # if bt depth > 0, needs to apply backtracking
    # everytime you pick a symbol, add it to excluded - if cannot pick: random pick
    if bt_depth_true:
        excluded_symbols: dict[int, list[str]] = {i: [] for i in range(bt_depth_true)}
        k = 0
        n_iter_btd = 0
        array_temp: np.ndarray[tuple[int, int], np.dtype[Any]] = array.copy()

        while k >= 0 and k < bt_depth_true and n_iter_btd < 100:
            n_iter_btd += 1
            try:
                pb_dist: dict[str, int | float] = extract_proba_from_tile(
                    array_temp, x, y + k, d_pe=d_proba_estimation, offsets=offsets
                )

                if pb_dist:
                    new_tile: str = generate_new_tile(
                        proba_dist=pb_dist,
                        all_symbols=all_symbols,
                        exclude=excluded_symbols[k],
                    )
                    excluded_symbols[k].append(new_tile)
                    array_temp[x + ti, y + tj + k] = new_tile
                    k += 1
                else:  # if pb_dist empty: unseen state
                    k -= 1

            except (
                AssertionError
            ):  # no tiles left for level k - need to try something else at k-1
                k -= 1  # backtracking

        if n_iter_btd == 100:  # btd got stuck in a back and forth loop
            logger.warning(msg="Backtracking stuck in loop! Random generation.")
            array[x + ti, y + tj] = generate_new_tile(
                proba_dist={}, all_symbols=all_symbols
            )
        if k == -1:  # no matter the tile used in base lvl it fails => random gen
            logger.warning(msg="Backtracking failed! Random generation.")
            array[x + ti, y + tj] = generate_new_tile(
                proba_dist={}, all_symbols=all_symbols
            )
        else:  # we made it through: tile at the top level works fine
            array[x + ti, y + tj] = array_temp[x + ti, y + tj]

    else:
        array[x + ti, y + tj] = generate_new_tile(
            proba_dist=extract_proba_from_tile(
                array, x, y, d_pe=d_proba_estimation, offsets=offsets
            ),
            all_symbols=all_symbols,
        )

    if verbose:
        logger.info(
            msg=f"New tile generated in position ({x + ti}, {y + tj}): "
            f"{array[x + ti, y + tj]}"
        )

    return array


def check_dpe_matches_mask(
    d_proba_estimation: dict[str, dict[str, int | float]],
    offsets: tuple[tuple[int, int], ...],
) -> None:
    """Asserts a DPE dict was trained with the mask about to be used with it.

    A mismatch used to pass silently: every lookup missed, `generate_new_tile`
    treated each miss as an unseen state and picked uniformly at random, and a
    run of pure noise looked like a successful generation.

    Args:
        d_proba_estimation: the loaded DPE dict.
        offsets: neighbourhood offsets of the mask generation will use.

    Raises:
        ValueError: the dict is empty, or its patterns are the wrong length.
    """
    if not d_proba_estimation:
        raise ValueError("Empty DPE dict - nothing to generate from.")

    pattern_len: int = len(next(iter(d_proba_estimation)))
    if pattern_len != len(offsets):
        raise ValueError(
            f"DPE patterns are {pattern_len} symbols long but the MdMC mask has "
            f"{len(offsets)} neighbourhood cells: this dict was trained with a "
            "different mask. Re-run dpe_generation with a matching --mdmc-matrix."
        )


def generate_room_data(
    d_proba_estimation: dict[str, dict[str, int | float]],
    room_size: tuple[int, int],
    backtracking_depth: int = 0,
    verbose: bool = False,
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]] | None = None,
    all_symbols: list[str] = ALL_TILES_AND_ENTITIES,
    border: str = "1",
) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
    """Samples a whole room, sweeping the window across it tile by tile.

    The sweep is row-major over an empty bordered room, so every window reads
    tiles that were either part of the starting border or sampled earlier in
    the same pass. Nothing here checks the room is playable - that is
    `level_generation`'s job.

    The same sweep serves both tile layers. Only two things differ, and both are
    arguments: the vocabulary an unseen pattern falls back on, and the symbol the
    empty room is bordered with. Background generation is
    `all_symbols=VALID_BG_TILES, border="0"`.

    Note that the target cell trails the window, so the first two rows and the
    first two columns are never written: rows 0 and column 0 keep the border, and
    row 1 and column 1 keep the empty room's void. On a foreground room that
    reads as the wall it is; on a background room it is a two-cell void margin
    along the top and left.

    Args:
        d_proba_estimation: the `{pattern: {symbol: probability}}` dict.
        room_size: `(width, height)` in tiles.
        backtracking_depth: lookahead passed to `update_room_array`.
        verbose: log every tile written.
        mdmc_matrix: the mask `d_proba_estimation` was trained with, defaulting
            to `DEFAULT_MDMC_MATRIX`.
        all_symbols: the layer's vocabulary, drawn from uniformly for an unseen
            pattern.
        border: the symbol the empty room is seeded with.

    Returns:
        The finished room, as a `(height, width)` array of tile symbols.

    Raises:
        ValueError: the mask is not 3x3, or does not match the dict.
    """
    matrix: np.ndarray[tuple[int, int], np.dtype[Any]] = (
        parse_mdmc_matrix(DEFAULT_MDMC_MATRIX) if mdmc_matrix is None else mdmc_matrix
    )
    if matrix.shape != (3, 3):
        raise ValueError(
            f"Room generation only supports 3x3 MdMC masks, got {matrix.shape} - "
            "the empty-room padding and the sweep bounds below assume that size."
        )
    offsets: tuple[tuple[int, int], ...] = mdmc_offsets(matrix)
    target: tuple[int, int] = mdmc_target_offset(matrix)
    check_dpe_matches_mask(d_proba_estimation, offsets)

    width, height = room_size[0], room_size[1]
    room_data: np.ndarray[tuple[int, int], np.dtype[Any]] = generate_empty_room_data(
        width, height, border=border
    )

    for x in range(height - 2):
        for y in range(width - 2):
            room_data = update_room_array(
                array=room_data,
                x=x,
                y=y,
                d_proba_estimation=d_proba_estimation,
                bt_depth=backtracking_depth,
                verbose=verbose,
                offsets=offsets,
                target=target,
                all_symbols=all_symbols,
            )

    return room_data


def generate_room_batch(
    n_rooms: int,
    output_dir: str,
    d_proba_estimation: dict[str, dict[str, int | float]],
    width: int = 40,
    height: int = 23,
    backtracking_depth: int = 0,
    verbose: bool = False,
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]] | None = None,
    all_symbols: list[str] = ALL_TILES_AND_ENTITIES,
    border: str = "1",
) -> None:
    """Samples `n_rooms` independent rooms and writes each one as a csv.

    The rooms are unrelated: no skeleton, no exits, no connectivity between
    them. Files land in `output_dir`, which is created if missing.

    Args:
        n_rooms: how many rooms to sample.
        output_dir: directory the csv files are written to.
        d_proba_estimation: the `{pattern: {symbol: probability}}` dict.
        width: room width, in tiles.
        height: room height, in tiles.
        backtracking_depth: lookahead passed to `update_room_array`.
        verbose: log every tile written.
        mdmc_matrix: the mask `d_proba_estimation` was trained with.
        all_symbols: the layer's vocabulary, see `generate_room_data`.
        border: the symbol the empty room is seeded with.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_rooms):
        room_data_temp: np.ndarray[tuple[int, int], np.dtype[Any]] = generate_room_data(
            d_proba_estimation,
            room_size=(width, height),
            backtracking_depth=backtracking_depth,
            verbose=verbose,
            mdmc_matrix=mdmc_matrix,
            all_symbols=all_symbols,
            border=border,
        )
        df_temp = pd.DataFrame(data=room_data_temp)
        df_temp.to_csv(
            path_or_buf=output_dir_path / f"room_{i}_generated_MdMC.csv",
            header=False,
            index=False,
            sep=";",
        )
        logger.info(
            msg=f"Successfully generated room {i}: saved to {output_dir}/room_{
                i
            }_generated_MdMC_{width}_{height}.csv"
        )


def generate_temp_room(
    d_exits: dict[str, list[list[int]]],
    d_proba_estimation: dict[str, dict[str, int | float]],
    backtracking_depth: int,
    room_size: tuple[int, int],
    origin: tuple[int, int] | tuple[int, ...],
    status: str | None,
    mdmc_matrix: np.ndarray[tuple[int, int], np.dtype[Any]] | None = None,
) -> Room:
    """Samples one room and fits it to the slot a skeleton reserved for it.

    Sampling only fills tiles; the exits the skeleton assigned still have to be
    carved back into the result, and the respawn points added, before the room
    can be placed in a level.

    Args:
        d_exits: the exits the skeleton assigned, per side.
        d_proba_estimation: the `{pattern: {symbol: probability}}` dict.
        backtracking_depth: lookahead passed to `update_room_array`.
        room_size: `(width, height)` in tiles.
        origin: the room's top-left corner in level coordinates.
        status: the skeleton's role for this room, e.g. start or end.
        mdmc_matrix: the mask `d_proba_estimation` was trained with.

    Returns:
        The room, with its exits carved and respawn points added.
    """
    # Imported here, not at module scope: `celeskeleton` imports `astar` from this
    # module, so a top-level import would make whichever of the two is imported
    # first fail on a partially initialized package. This is the only place in
    # `utils` that needs a `Room` at runtime, so the cycle costs one local import.
    from seleCte.celeskeleton.celeskeleton import Room as RoomImpl

    temp_room = RoomImpl(width=room_size[0], height=room_size[1], status=status)

    temp_data: np.ndarray[tuple[int, int], np.dtype[Any]] = generate_room_data(
        d_proba_estimation,
        room_size=room_size,
        backtracking_depth=backtracking_depth,
        mdmc_matrix=mdmc_matrix,
    )

    temp_room.set_data(temp_data)
    temp_room.set_origin(x=origin[0], y=origin[1])
    temp_room.set_exits(d_exits)
    temp_room.create_exits_in_matrix()
    temp_room.add_respawn_points()

    return temp_room


# Helper functions for playability module
class Node:
    """A node class for A* Pathfinding."""

    def __init__(
        self,
        parent: Node | None,
        position: tuple[int, int],
        direction_cost_factor: float = 1.0,
    ) -> None:
        """Creates a node at `position`, reached from `parent`.

        Args:
            parent: the node this one was expanded from, `None` for the start.
            position: `(row, column)` in the room.
            direction_cost_factor: per-move weight for the step that got here,
                which is how the search is biased towards platformer-plausible
                movement rather than treating every direction alike.
        """
        self.position: tuple[int, int] = position
        self.parent: Node | None = parent
        self.direction_cost_factor: float = direction_cost_factor

        self.g: float = 0.0  # Cost from start to node
        self.h: float = 0.0  # Heuristic cost from node to end
        self.f: float = 0.0  # Total cost (g + h)

    def __eq__(self, other: object) -> bool:
        """Compares nodes by position, so a cell is one node however reached."""
        return isinstance(other, Node) and self.position == other.position

    def __repr__(self) -> str:
        """Returns the position and the three A* costs, for debugging."""
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other: Node) -> bool:
        """Orders nodes by total cost, so the heap pops the cheapest first."""
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other: Node) -> bool:
        """Orders nodes by total cost, the mirror of `__lt__`."""
        return self.f > other.f


def return_path(current_node: Node) -> list[tuple[int, int]]:
    """Walks the parent chain back from a node to rebuild the path to it.

    Args:
        current_node: the node the search ended on.

    Returns:
        The positions from start to `current_node`, in order.
    """
    path = []
    current: Node | None = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(
    room: Room,
    maze: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    allow_diagonal_movement: bool = True,
    max_iterations: int | None = None,
) -> list[tuple[int, int]]:
    """Returns - if it exists - a path from `start` to `end` through `maze`.

    **Do not replace or edit this when the flood-based playability model lands.**
    It is deliberately kept as the comparison baseline: the replacement has to be
    measured against it with the movement model held constant, so the new search
    belongs beside this function rather than on top of it.

    What it models is 8-connected movement through open cells, with weighted
    direction costs. There is no gravity, no jump or dash reach, no stamina and no
    wall-grab, so "a path exists" is close to always true and says very little
    about Celeste. It certifies rooms that cannot be played and rejects rooms that
    are fine.

    It is also slow, and not where you would guess: the cost is the linear rescans
    of `closed_list` and `open_list` once per child, not the search. One check at a
    20k iteration cap profiles at 123.6 s with 62% of that inside `Node.__eq__`.
    Sets of positions remove essentially all of it. `get_min_dist_to_nle` is 2.2%
    and is not worth optimising.

    Args:
        room: the room the maze describes, used to score how close a step
            stays to solid ground.
        maze: the room as a cost grid, where `0` marks a walkable cell.
        start: `(row, column)` to search from.
        end: `(row, column)` to search for.
        allow_diagonal_movement: also expand the four diagonal neighbours.
        max_iterations: give up after this many expansions, defaulting to the
            cell count of the maze.

    Returns:
        The positions from `start` to `end`, or an empty list if no path was
        found before the iteration limit.
    """
    nb_iter = 0

    # Create start and end node
    start_node = Node(parent=None, position=start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(parent=None, position=end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list: list[Node] = []
    closed_list: list[Node] = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Adding a stop condition if max_iterations is None
    outer_iterations = 0
    max_iter: int = max_iterations or len(maze[0]) * len(maze)

    # Which squares do we search
    adjacent_squares: tuple[tuple[int, int], ...]
    direction_cost: tuple[float, ...]
    if allow_diagonal_movement:
        adjacent_squares = (
            (0, -1),  # left
            (0, 1),  # right
            (-1, 0),  # up
            (1, 0),  # down
            (-1, -1),  # up-left
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (1, 1),  # down-right
        )
        direction_cost = (0.1, 0.1, 0.05, 0, 0.5, 0.5, 0.01, 0.01)
        # direction_cost = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        adjacent_square_pick_index = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0))
        direction_cost = (1.0, 1.0, 1.0, 1.0)
        adjacent_square_pick_index = [0, 1, 2, 3]

    # Loop until you find the end
    while len(open_list) > 0:
        nb_iter += 1
        # Randomize the order of the adjacent_squares_pick_index to avoid a
        # decision making bias
        random.shuffle(x=adjacent_square_pick_index)
        outer_iterations += 1

        if outer_iterations > max_iter:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            logger.warning(msg=f"Iteration limit exceeded - stopped at iter {nb_iter}")
            return []

        # Get the current node
        current_node: Node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            logger.info(msg=f"Found a path in {nb_iter} iterations")
            return return_path(current_node)

        # Generate children
        children: list[Node] = []

        for pick_index in adjacent_square_pick_index:
            new_position: tuple[int, int] = adjacent_squares[pick_index]
            direction_cost_factor: float = direction_cost[pick_index]

            # Get node position
            node_position: tuple[int, int] = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1],
            )

            # Make sure within range
            if (
                node_position[0] > (len(maze) - 1)
                or node_position[0] < 0
                or node_position[1] > (len(maze[len(maze) - 1]) - 1)
                or node_position[1] < 0
            ):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(
                parent=current_node,
                position=node_position,
                direction_cost_factor=direction_cost_factor,
            )

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if (
                len(
                    [
                        closed_child
                        for closed_child in closed_list
                        if closed_child == child
                    ]
                )
                > 0
            ):
                continue

            # Create the f, g, and h values
            # x, y = child.position
            # local_path = [(x, k) for k in range(y-3, y+4) if 0 <= k < len(maze[0])]
            child.g = current_node.g + (
                child.direction_cost_factor
                + get_min_dist_to_nle(room, pos=child.position) / 10
            ) * len(maze[0]) * len(maze)
            # child.g = current_node.g + direction_cost_factor
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                (child.position[1] - end_node.position[1]) ** 2
            )
            child.f = child.g + child.h

            # Child is already in the open list
            if (
                len(
                    [
                        open_node
                        for open_node in open_list
                        if child.position == open_node.position
                        and child.g > open_node.g
                    ]
                )
                > 0
            ):
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    logger.warning(msg="No path to destination")
    return []


## Evaluation time
def create_paths_heatmap(
    room: Room, nb_paths: int, c_map: str = "viridis", max_iter: int = 0
) -> Styler:
    """Counts how often repeated A* runs cross each cell, and colours the room.

    A* shuffles its expansion order, so each run may take a different route.
    Sampling several and stacking them shows which parts of a room the search
    actually uses, rather than the one path a single run happened to pick.
    Solid tiles are counted as `-1` and drawn flat grey.

    Args:
        room: the room to trace paths through.
        nb_paths: how many A* runs to stack.
        c_map: matplotlib colormap name for the visit counts.
        max_iter: iteration limit per run, `0` for the A* default.

    Returns:
        The visit counts, styled.
    """
    l_path: list[list[tuple[int, int]]] = [
        room.extract_room_path(max_iterations=max_iter) for _ in range(nb_paths)
    ]

    heatmap_data: np.ndarray[tuple[int, int], np.dtype[Any]] = np.zeros(
        room.data.shape, dtype=int
    )
    for pos in [
        (x, y)
        for x, y in zip(
            np.where(room.data != "0")[0], np.where(room.data != "0")[1], strict=False
        )
    ]:
        heatmap_data[pos] = -1

    for path in l_path:
        for path_pos in path:
            heatmap_data[path_pos] += 1

    df = pd.DataFrame(data=heatmap_data)

    max_value: int = df.max().max()
    min_value = 0
    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap: Colormap = plt.get_cmap(name=c_map)

    # Step 3: Function to apply the color scale
    def color_scale(val: int) -> str:
        if val == -1:
            color = (0.6, 0.6, 0.6, 1.0)
        elif val == 0:
            color = (0.1, 0.1, 0.1, 1.0)
        else:
            color = cmap(X=norm(value=val))
        return f"background-color: {mcolors.rgb2hex(c=color)}"

    # Step 4: Apply the color scale to the DataFrame
    # Same as `visualize_room`: the frame here only ever holds visit counts.
    return df.style.map(func=lambda val: color_scale(val=cast("int", val)))


def get_interest_space(
    array: np.ndarray[tuple[int, int], np.dtype[Any]],
    path: list[tuple[int, int]],
    sensibility: int,
) -> list[tuple[int, int]]:
    """Returns the zone of interest: the cells a path passes close to.

    The metrics score a room along the route the player would take rather than
    over the whole room, and this is the neighbourhood that defines "along".
    Cells falling outside the room are dropped; cells covered by more than one
    path position are repeated, so the counts stay proportional to how much of
    the path runs near them.

    Args:
        array: the room, used for its bounds.
        path: the positions the path visits.
        sensibility: half-width of the square window kept around each of them.

    Returns:
        The in-bounds cells of the zone of interest, with duplicates.
    """
    xmax, ymax = array.shape
    l_interest_area, potential_neighbors = [], []
    for x, y in path:
        potential_neighbors.extend(
            [
                (x + i, y + j)
                for i in range(-1 * sensibility, sensibility + 1)
                for j in range(-1 * sensibility, sensibility + 1)
            ]
        )
    for x_pot, y_pot in potential_neighbors:
        if x_pot >= 0 and x_pot < xmax and y_pot >= 0 and y_pot < ymax:
            l_interest_area.append((x_pot, y_pot))
    return l_interest_area


def visualize_room_path(
    room: Room,
    sensibility: int,
    max_iter: int = 0,
) -> Styler | None:
    """Draws one A* path over a room, with the zone of interest around it.

    Path cells are marked `"p"` and zone-of-interest cells `"e"`, overwriting
    whatever tile was there, so this is for looking at rooms rather than for
    producing data.

    Args:
        room: the room to trace a path through.
        sensibility: half-width passed to `get_interest_space`.
        max_iter: iteration limit for A*, `0` for the default.

    Returns:
        The styled room, or `None` if no path was found.
    """
    path: list[tuple[int, int]] = room.extract_room_path(max_iterations=max_iter)

    if not path:
        logger.warning(msg="No path has been found for this room.")
        return None

    room_data_path: list[tuple[int, int]] = get_interest_space(
        array=room.data, path=path, sensibility=sensibility
    )
    df_visu = pd.DataFrame(room.data)

    for tile_ in room_data_path:
        df_visu.loc[tile_] = "e"
    for tile in path:
        df_visu.loc[tile] = "p"

    return visualize_room(df=df_visu)


def extract_entity_coords(room: Room, symbol: str) -> list[tuple[int, int]]:
    """Returns every position in a room holding a given symbol.

    Args:
        room: the room to search.
        symbol: the tile or entity symbol to look for.

    Returns:
        The matching `(row, column)` positions.
    """
    return [
        (x, y)
        for x, y in zip(
            np.where(room.data == symbol)[0],
            np.where(room.data == symbol)[1],
            strict=False,
        )
    ]


def extract_non_lethal_entities_position(room: Room) -> list[tuple[int, int]]:
    """Returns the positions of everything the player can safely use.

    Solid tiles count alongside the non-lethal entities: standing on ground is
    as good as landing on a spring as far as the metrics are concerned.

    Args:
        room: the room to search.

    Returns:
        The matching `(row, column)` positions.
    """
    nle_pos: list[tuple[int, int]] = []
    l_nle: list[str] = [*NON_LETHAL_ENTITIES, "1"]

    for entity in l_nle:
        nle_pos.extend(extract_entity_coords(room, symbol=entity))

    return nle_pos


def extract_non_lethal_entities_position_diff(room: Room) -> list[tuple[int, int]]:
    """Same as `extract_non_lethal_entities_position`, but counting void too.

    Adding `"0"` makes the complement of this set the lethal entities alone,
    which is what the difficulty metric measures density over.

    Args:
        room: the room to search.

    Returns:
        The matching `(row, column)` positions.
    """
    nle_pos: list[tuple[int, int]] = []
    l_nle: list[str] = [*NON_LETHAL_ENTITIES, "0", "1"]

    for ent in l_nle:
        nle_pos.extend(extract_entity_coords(room, symbol=ent))

    return nle_pos


def hole_presence(room: Room, pos: tuple[int, int]) -> bool:
    """Return True if there is a hole below the current position."""
    pot_hole = room.data[pos[0] :, pos[1]]
    return not ("1" in pot_hole or "_" in pot_hole or "D" in pot_hole)


def danger_presence(room: Room, pos: tuple[int, int]) -> bool:
    """Returns True if falling from a position lands the player on a hazard.

    Only the first non-empty tile below matters, since that is what a fall
    stops on. A column of pure void counts as dangerous.

    Args:
        room: the room to look in.
        pos: the `(row, column)` to fall from.

    Returns:
        Whether the landing is lethal.
    """
    pot_danger = room.data[pos[0] :, pos[1]]
    for sym in pot_danger:
        if sym == "0":
            continue
        return bool(sym == "^" or sym == "S")
    return True


def danger_density_normalized(room: Room, path: list[tuple[int, int]]) -> float:
    """Return the density of danger within a path found by A*, normalized."""
    l_danger: list[bool] = [danger_presence(room, pos) for pos in path]
    return sum(l_danger) / len(l_danger)


def evaluate_room_difficulty(
    room: Room,
    path: list[tuple[int, int]],
    sensibility: int,
) -> tuple[float, float, float]:
    """Rates a room's difficulty from hazards, emptiness and scarcity.

    Half-half contribution atm, can be weighted differently.
    NB: density_diff + density_nle = 1.

    Args:
        room: the room to score.
        path: the A* path to score along.
        sensibility: half-width passed to `get_interest_space`.

    Returns:
        The share of the path that falls onto a hazard, the density of lethal
        entities and void in the zone of interest, and how sparse the safe
        ground is there.
    """
    holes_diff: float = danger_density_normalized(room, path)
    # LE + void density: basically all that is not NLE
    zone_of_interest: list[tuple[int, int]] = get_interest_space(
        array=room.data, path=path, sensibility=sensibility
    )
    entities_pos: list[tuple[int, int]] = extract_non_lethal_entities_position_diff(
        room
    )
    entities_pos_nle: list[tuple[int, int]] = extract_non_lethal_entities_position(room)

    nb_diff_zoi: int = len(zone_of_interest) - len(
        [pos for pos in entities_pos if pos in zone_of_interest]
    )
    nb_nle_zoi: int = len([pos for pos in entities_pos_nle if pos in zone_of_interest])

    local_density_diff: float = nb_diff_zoi / len(zone_of_interest)
    scarcity: float = len(zone_of_interest) / nb_nle_zoi

    return holes_diff, local_density_diff, scarcity


def get_entropy_data(room_data: np.ndarray[tuple[int, int], np.dtype[Any]]) -> float:
    """Returns the Shannon entropy of a room's tile distribution, in base 10.

    A room using one symbol everywhere scores 0; the more evenly the room
    spreads across its vocabulary, the higher it scores. This measures variety
    only - where the symbols sit is not considered.

    Args:
        room_data: the room, as an array of tile symbols.

    Returns:
        The entropy.
    """
    d_symbols: dict[str, int] = {}
    entropy = 0
    room_number_of_tiles: int = room_data.size
    for line in room_data:
        for j in line:
            if j not in d_symbols:
                d_symbols[j] = 1
            else:
                d_symbols[j] += 1

    for key in d_symbols:
        p: float = d_symbols[key] / room_number_of_tiles
        entropy += -p * np.log10(p)

    return entropy


def evaluate_room_interestingness(
    room: Room,
    path: list[tuple[int, int]],
    sensibility: int,
) -> tuple[float, float, float]:
    """Density of tiles + non-lethal entities.

    Return the density of tiles + NLE in the whole room then
    the same quantity computed in the ZOI, noted as d_tot and
    s_int.
    0 < d_nle < 1: 0 - empty room besides LE; 1 - room full
    0 < s_int < 1: same but in ZOI.

    Args:
        room: the room to score.
        path: the A* path to score along.
        sensibility: half-width passed to `get_interest_space`.

    Returns:
        The room-wide density, the zone-of-interest density, and the room's
        tile entropy.
    """
    zone_of_interest: list[tuple[int, int]] = get_interest_space(
        array=room.data, path=path, sensibility=sensibility
    )
    entities_pos: list[tuple[int, int]] = extract_non_lethal_entities_position(room)

    nb_nle_tot: int = len(entities_pos)
    nb_nle_zoi: int = len([pos for pos in entities_pos if pos in zone_of_interest])

    density_nle: float = nb_nle_tot / room.data.size
    interestingness_score: float = nb_nle_zoi / len(zone_of_interest)

    return density_nle, interestingness_score, get_entropy_data(room_data=room.data)


def get_coords_around_pos(
    room: Room,
    pos: tuple[int, int],
    dist: int,
) -> list[tuple[int, int]]:
    """Returns the ring of cells exactly `dist` away from a position.

    The ring is the square outline at Chebyshev distance `dist`, clipped to the
    room, and kept to rows at or below `pos` - `get_min_dist_to_nle` grows it
    outwards to find the nearest safe ground under the player, so cells above
    are not what it is looking for.

    Args:
        room: the room, used for its bounds.
        pos: the `(row, column)` to search around.
        dist: the ring's radius, in tiles.

    Returns:
        The in-bounds cells of the ring.
    """

    def is_valid_coordinate(
        array: np.ndarray[tuple[int, int], np.dtype[Any]], x: int, y: int
    ) -> bool:
        max_x: int = len(array)
        max_y: int = len(array[0]) if max_x > 0 else 0
        return 0 <= x < max_x and 0 <= y < max_y

    x, y = pos
    list_pos: list[tuple[int, int]] = []
    for side in (-1, 1):
        list_pos.extend([(x + side * dist, y + k) for k in range(-1 * dist, dist + 1)])
        list_pos.extend([(x + k, y + side * dist) for k in range(-1 * dist + 1, dist)])

    return [
        (a, b)
        for (a, b) in list_pos
        if is_valid_coordinate(array=room.data, x=a, y=b) and a >= x
    ]


def get_min_dist_to_nle(
    room: Room,
    pos: tuple[int, int],
) -> int:
    """Returns the distance from a position to the nearest safe ground.

    Rings are searched outwards until one contains a solid tile or a non-lethal
    entity. A* uses this to penalise steps that drift away from anything the
    player could stand on.

    Args:
        room: the room to search.
        pos: the `(row, column)` to measure from.

    Returns:
        The Chebyshev distance, `0` if `pos` itself is safe ground.

    Raises:
        ValueError: the room holds no safe ground at all, so it could not be
            played whatever path is taken.
    """
    l_nle: list[str] = [*NON_LETHAL_ENTITIES, "1"]
    dist = 1
    if room.data[pos] in l_nle:
        return 0
    while dist < max(room.data.shape):
        l_pos_to_check: list[tuple[int, int]] = get_coords_around_pos(room, pos, dist)
        for pot_pos in l_pos_to_check:
            if room.data[pot_pos] in l_nle:
                return dist
        dist += 1
    # if not found: room has not a single NLE: not playable anyways
    raise ValueError("The room currently investigated has not a single NLE!")


def evaluate_astar_path(
    room: Room,
    path: list[tuple[int, int]],
) -> float:
    """Avg. distance to tiles + NL entities."""
    l_dist_path_to_nle: list[int] = []
    for path_pos in path:
        l_dist_path_to_nle.append(get_min_dist_to_nle(room, path_pos))

    return sum(l_dist_path_to_nle) / len(l_dist_path_to_nle)


def variance_astar_path(
    room: Room,
    path: list[tuple[int, int]],
) -> float:
    """Return the population variance of path distances to safe ground.

    Measure each position to the nearest solid tile or non-lethal entity with
    ``get_min_dist_to_nle``. These are the same distances that
    ``evaluate_astar_path`` averages.

    Args:
        room: The room containing the path.
        path: Path positions as ``(row, column)`` pairs.

    Returns:
        The population variance of the Chebyshev distances along the path.
    """
    l_dist_path_to_nle: list[int] = []
    for path_pos in path:
        l_dist_path_to_nle.append(get_min_dist_to_nle(room, path_pos))

    return np.var(a=l_dist_path_to_nle)
