"""Skeleton stage of the level generator: room rectangles and how they connect.

A `Celeskeleton` is the coarse geometry of a level, before any tile content is
authored: a mapping of room name to `Room`, where each room is an axis-aligned
rectangle placed on one integer grid shared by the whole level. `Room.origin`
is the bottom-left corner `[x, y]`, `Room.size` is `[width, height]`, and
`Room.exits` records the openings on each side as
`{"up" | "down" | "left" | "right": [[c1, c2], ...]}`. Exit coordinates are
absolute grid positions, not offsets from the room, which is why `set_origin`
shifts them along with the room.

Rooms also carry `data`, a `(height, width)` array of single-character tile
codes. Later stages of the pipeline author its contents; this module only draws
the borders, punches the exits through them, and places the player spawn and
the crystal heart in the starting and ending rooms.

The module covers four concerns:

- Construction: `Room.connect` juxtaposes two rooms and opens a shared exit
  between them, and `generate_celeskeleton` chains that into a full level -
  from a corridor (`p = 0`) to a labyrinth (`p` close to 1).
- Playability: `Room.is_playable_room` and `Room.is_valid_special` run A* over
  the tile grid to check that a room's exits can all reach one another.
- Persistence: `Celeskeleton.save` writes a `data.json` following the
  `SkeletonJSON` schema plus one csv per room, and `load_data_to_celeskeleton`
  reads that pair back into a skeleton.
- Visualization: `Celeskeleton.show_skeleton` plots the layout with matplotlib.
"""

import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from seleCte.utils import astar

logger: logging.Logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

# Colors used by Celeskeleton.show_skeleton
ROOM_EDGE_COLOR: str = "#37474f"
EXIT_COLOR: str = "#e53935"
STATUS_COLORS: dict[str, str] = {"start": "#00c853", "end": "#d500f9"}

# TODO: adapt is playable to 1-exit rooms for space assertion
# once the strawberry spawner is done (with strawberry as exit point)

# The side a neighbour presents back, per side. `Room.connect` always opens the
# opening on both rooms at once - the room below gets an `"up"` exit and the room
# above the matching `"down"` one, at the same coordinates - so a pair of rooms
# sharing an opening can be recognised afterwards from their exits alone.
OPPOSITE_SIDE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


class RoomJSON(TypedDict):
    """On-disk shape of a single room, as written by `Room.save`/`Celeskeleton.save`.

    JSON has no tuple type: `origin` and `size` are 2-element *lists* once they
    come back from `json.load`, so that is what this schema promises. Anything
    that needs a `tuple[int, int]` converts on the way in (see
    `json_to_skeleton`), never by annotating the raw payload.
    """

    origin: list[int]
    size: list[int]
    exits: dict[str, list[list[int]]]


# A whole `data.json`: room name -> room payload.
SkeletonJSON = dict[str, RoomJSON]


class Room:
    """Contains all the information related to a room in Celeste."""

    def __init__(
        self,
        width: int = 1,
        height: int = 1,
        origin_x: int = 0,
        origin_y: int = 0,
        status: str | None = None,
    ) -> None:
        """Init Room object."""
        if width <= 0 or height <= 0:
            raise ValueError("Width and height have to be positive!")
        self.size: tuple[int, int] = (width, height)
        self.origin: tuple[int, int] = (
            origin_x,
            origin_y,
        )
        self.exits: dict[str, list[list[int]]] = {
            "up": [],
            "down": [],
            "left": [],
            "right": [],
        }
        self.data: np.ndarray[tuple[int, int], np.dtype[Any]] = np.zeros(
            shape=(self.size[1], self.size[0]), dtype=int
        ).astype(dtype=str)
        self.status: str | None = status

    def get_status(self) -> str | None:
        """Gets status of a given room."""
        return self.status

    def get_size(self) -> tuple[int, int]:
        """Gets size of a given room."""
        return self.size

    def get_origin(self) -> tuple[int, int]:
        """Gets origin within a given room."""
        return self.origin

    def get_exits(self) -> dict[str, list[list[int]]]:
        """Gets the exits within a given room."""
        return self.exits

    def description(self) -> str:
        """Description of a given room."""
        return f"Bottom-left corner position: {self.origin}\nRoom size: {
            self.size
        }\nExits: {self.exits}\nData: {self.data}\nStatus: {self.status}"

    def get_tr_corner(self) -> list[int]:
        """Returns coordinates of top right corner of a room."""
        return [self.origin[0] + self.size[0], self.origin[1] + self.size[1]]

    def get_corners(self) -> list[int]:
        """Returns coordinates of bottom left and top right corner as a list."""
        return list(self.get_origin()) + list(self.get_tr_corner())

    def clear_exits(self) -> None:
        """Clears exits dict."""
        self.exits = {
            "up": [],
            "down": [],
            "left": [],
            "right": [],
        }

    def set_status(self, status: str | None) -> None:
        """Sets status within a given room."""
        if status == "start" or status == "end":
            self.status = status
        else:
            raise ValueError("Room can only have start or end status, or None.")

    def set_size(self, width: int, height: int) -> None:
        """Setter to modify size of a room.

        Might interfer with exits positions so you can
        only change size of a room when there are no exits.
        """
        exits: dict[str, list[list[int]]] = self.get_exits()
        for side in exits:
            if exits[side]:  # at least one exit
                raise ValueError(
                    f"At least one exit detected {side}, please use "
                    "the clear_exits method if you want to change the size of the room."
                )
        if width <= 0 or height <= 0:
            raise ValueError("Width and height have to be positive!")
        self.size = (width, height)

    def set_origin(self, x: int, y: int) -> None:
        """Setter to modify the origin point (bottom left corner) of a room.

        Takes into account exits and move them accordingly.
        """
        old_x, old_y = self.get_origin()
        delta: list[int] = [x - old_x, y - old_y]
        self.origin = (x, y)
        sides: list[str] = ["up", "left", "down", "right"]
        for k in range(4):
            for exit in self.exits[sides[k]]:
                exit[0], exit[1] = exit[0] + delta[k % 2], exit[1] + delta[k % 2]

    def set_exits(self, d_exits: dict[str, list[list[int]]]) -> None:
        """Sets the exits within a given room."""
        self.exits = d_exits

    def set_data(self, data: np.ndarray[tuple[int, int], np.dtype[Any]]) -> None:
        """Sets the tiles within a given room."""
        if data.shape[0] == self.get_size()[1] and data.shape[1] == self.get_size()[0]:
            self.data = data
        else:  # mismatch in the size - should raise an error
            raise ValueError(
                f"Can't assign data of shape {data.shape} to a room of shape {
                    (self.get_size()[1], self.get_size()[0])
                }!"
            )

    def add_exit(self, side: str, c1: int, c2: int) -> None:
        """Generate exits based on coordinates and side."""
        if c1 >= c2:
            raise ValueError("Error: make sure coordinates are ordered and non equal")
        if side not in self.exits:
            raise ValueError(f"Error: Select a side among {list(self.exits)}")
        if side in ["up", "down"]:
            if (
                c1 >= self.get_origin()[0] + 1
                and c2 <= self.get_origin()[0] + self.get_size()[0] - 1
            ):
                self.exits[side].append([c1, c2])
            else:
                raise ValueError(
                    f"Exit out of bounds {
                        (
                            self.get_origin()[0] + 1,
                            self.get_origin()[0] + self.get_size()[0] - 1,
                        )
                    }!"
                )
        else:
            if (
                c1 >= self.get_origin()[1] + 1
                and c2 <= self.get_origin()[1] + self.get_size()[1] - 1
            ):
                self.exits[side].append([c1, c2])
            else:
                raise ValueError(
                    f"Exit out of bounds {
                        (
                            self.get_origin()[1] + 1,
                            self.get_origin()[1] + self.get_size()[1] - 1,
                        )
                    }!"
                )

    def generate_exit(self, side: str, size: int) -> None:
        """Generate a random exit simply with size + side."""
        if side in ["up", "down"]:
            c: int = np.random.randint(
                low=self.get_origin()[0] + 1,
                high=self.get_origin()[0] + self.get_size()[0] - size,
            )
            self.add_exit(side, c1=c, c2=c + size)
        elif side in ["left", "right"]:
            c = np.random.randint(
                low=self.get_origin()[1] + 1,
                high=self.get_origin()[1] + self.get_size()[1] - size,
            )
            self.add_exit(side, c1=c, c2=c + size)
        else:
            raise ValueError(f"Error: Select a side among {list(self.exits)}")

    def exits_to_list(self) -> list[tuple[int, int, int, int]]:
        """Returns the list of coordinates of the exits of the room as quadruplets."""
        l_exits: list[tuple[int, int, int, int]] = []
        sides: list[str] = ["left", "down", "right", "up"]
        base_coord: list[int] = self.get_corners()
        for k in range(4):
            for exit in self.exits[sides[k]]:
                x0: int = base_coord[k] * ((k + 1) % 2) + (k % 2) * exit[0]
                x1: int = base_coord[k] * ((k + 1) % 2) + (k % 2) * exit[1]
                y0: int = base_coord[k] * (k % 2) + ((k + 1) % 2) * exit[0]
                y1: int = base_coord[k] * (k % 2) + ((k + 1) % 2) * exit[1]
                l_exits.append((x0, y0, x1, y1))

        return l_exits

    def is_overlapping(self, room: Room) -> bool:
        """Detects whether room is overlapping with another rectangle rect."""
        rect_1: list[int] = self.get_corners()
        rect_2: list[int] = room.get_corners()
        return not (
            (rect_1[0] >= rect_2[2])
            or (rect_1[2] <= rect_2[0])
            or (rect_1[3] <= rect_2[1])
            or (rect_1[1] >= rect_2[3])
        )

    def faces(self, room: Room, side: str) -> bool:
        """Returns whether `room` sits flush against this one on `side`.

        Adjacency only, with no regard for exits: the two rooms share the border
        line, which is the precondition for them being able to share an opening
        through it.

        Args:
            room: the room to test against.
            side: which side of *this* room to look at.

        Returns:
            Whether the two borders coincide.

        Raises:
            ValueError: `side` is not one of the four sides.
        """
        x0, y0 = self.get_origin()
        x1, y1 = self.get_tr_corner()
        room_x0, room_y0 = room.get_origin()
        room_x1, room_y1 = room.get_tr_corner()

        if side == "up":
            return y1 == room_y0
        if side == "down":
            return y0 == room_y1
        if side == "left":
            return x0 == room_x1
        if side == "right":
            return x1 == room_x0
        raise ValueError(f"Error: Select a side among {list(self.exits)}")

    def is_connected_to(self, room: Room) -> bool:
        """Returns whether an exit of this room opens directly onto `room`.

        Two rooms are connected when they are flush against each other and hold
        the same exit segment on the two sides facing one another. That pair is
        exactly what `connect` writes, and because exit coordinates are absolute
        rather than room-relative, matching segments cannot be a coincidence
        between rooms that are not the two ends of one opening.

        Being juxtaposed is not enough on its own: `generate_celeskeleton` packs
        rooms tightly, so rooms routinely touch without anything linking them.

        Args:
            room: the room to test against.

        Returns:
            Whether the two rooms share at least one opening.
        """
        for side, opposite in OPPOSITE_SIDE.items():
            if not self.faces(room, side=side):
                continue
            own_exits: set[tuple[int, ...]] = {tuple(e) for e in self.exits[side]}
            facing_exits: set[tuple[int, ...]] = {
                tuple(e) for e in room.exits[opposite]
            }
            if own_exits & facing_exits:
                return True

        return False

    def connect(self, room: Room, side: str, skeleton: Celeskeleton) -> None:
        """Move self so that self is juxtaposed to room on the selected side."""
        x0, y0 = room.get_origin()

        if side == "up":
            self.set_origin(
                x=np.random.randint(
                    low=x0 + 3 - self.get_size()[0], high=x0 + room.get_size()[0] - 2
                ),
                y=y0 + room.get_size()[1],
            )
            ovl: bool = skeleton.is_overlapping_with(room=self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = (
                    max(x0, x0_bis),
                    min(x0 + room.get_size()[0], x0_bis + self.get_size()[0]),
                )
                size: int = np.random.randint(low=2, high=range_max - range_min - 1)
                c: int = np.random.randint(low=range_min + 1, high=range_max - size - 1)
                self.add_exit(side="down", c1=c, c2=c + size)
                room.add_exit(side="up", c1=c, c2=c + size)

        elif side == "down":
            self.set_origin(
                x=np.random.randint(
                    low=x0 + 3 - self.get_size()[0], high=x0 + room.get_size()[0] - 2
                ),
                y=y0 - self.get_size()[1],
            )
            ovl = skeleton.is_overlapping_with(room=self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = (
                    max(x0, x0_bis),
                    min(x0 + room.get_size()[0], x0_bis + self.get_size()[0]),
                )
                size = np.random.randint(low=2, high=range_max - range_min - 1)
                c = np.random.randint(low=range_min + 1, high=range_max - size - 1)
                self.add_exit(side="up", c1=c, c2=c + size)
                room.add_exit(side="down", c1=c, c2=c + size)

        elif side == "right":
            self.set_origin(
                x=x0 + room.get_size()[0],
                y=np.random.randint(
                    low=y0 + 3 - self.get_size()[1], high=y0 + room.get_size()[1] - 2
                ),
            )
            ovl = skeleton.is_overlapping_with(room=self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = (
                    max(y0, y0_bis),
                    min(y0 + room.get_size()[1], y0_bis + self.get_size()[1]),
                )
                size = np.random.randint(low=2, high=range_max - range_min - 1)
                c = np.random.randint(low=range_min + 1, high=range_max - size - 1)
                self.add_exit(side="left", c1=c, c2=c + size)
                room.add_exit(side="right", c1=c, c2=c + size)

        elif side == "left":
            self.set_origin(
                x=x0 - self.get_size()[0],
                y=np.random.randint(
                    low=y0 + 3 - self.get_size()[1], high=y0 + room.get_size()[1] - 2
                ),
            )
            ovl = skeleton.is_overlapping_with(room=self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = (
                    max(y0, y0_bis),
                    min(y0 + room.get_size()[1], y0_bis + self.get_size()[1]),
                )
                size = np.random.randint(low=2, high=range_max - range_min - 1)
                c = np.random.randint(low=range_min + 1, high=range_max - size - 1)
                self.add_exit(side="right", c1=c, c2=c + size)
                room.add_exit(side="left", c1=c, c2=c + size)
        else:
            raise ValueError("Selected side incorrect")

    def _local_row(self, y_abs: int) -> int:
        """Converts an absolute skeleton y coordinate to a row index into `data`.

        Row 0 is the room's *top* edge - `create_exits_in_matrix` carves an
        `"up"` exit into `data[0]`, and `data_to_level.jl` positions a room by
        its top-left corner - while the skeleton's y axis points *up* from
        `origin`. The two run opposite ways, so this flips.

        The flip is what makes an opening land at the same absolute height in
        both rooms that share it: the result reduces to a function of `y_abs`
        alone, independent of the room's origin and height.

        Args:
            y_abs: a skeleton y coordinate, as stored in `exits`.

        Returns:
            The matching row index into `data`.
        """
        return (self.size[1] - 1) - (y_abs - self.origin[1])

    def _local_col(self, x_abs: int) -> int:
        """Converts an absolute skeleton x coordinate to a column index into `data`.

        Both x axes point right, so unlike `_local_row` there is nothing to flip.

        Args:
            x_abs: a skeleton x coordinate, as stored in `exits`.

        Returns:
            The matching column index into `data`.
        """
        return x_abs - self.origin[0]

    def get_starting_ending_points(self) -> list[tuple[int, int]]:
        """Returns one point per exit, where a path through the room starts or ends.

        Exit coordinates are absolute skeleton positions, so each is converted
        through `_local_row`/`_local_col`. Getting that conversion wrong does not
        raise: it yields a point outside the room, A* then searches for a cell
        that does not exist, and every playability check on the room fails for
        good rather than failing loudly.

        Returns:
            One `(row, column)` per exit, on the border that exit opens through.
        """
        l_points: list[tuple[int, int]] = []
        for exit in self.exits["up"]:
            l_points.append((0, self._local_col(int(np.mean(a=exit)))))
        for exit in self.exits["down"]:
            l_points.append((self.size[1] - 1, self._local_col(int(np.mean(a=exit)))))
        for exit in self.exits["left"]:
            l_points.append((self._local_row(int(np.mean(a=exit))), 0))
        for exit in self.exits["right"]:
            l_points.append((self._local_row(int(np.mean(a=exit))), self.size[0] - 1))

        return l_points

    def extract_astar_ready_data(self) -> np.ndarray[tuple[int, int], np.dtype[Any]]:
        """Make a room Celeste-A* ready.

        Translate entites Madz can go through by swapping 0s instead, 1s otherwise.
        """
        # TODO: what about one-way tiles?
        df_rdy_data = pd.DataFrame(self.data)
        df_rdy_data = df_rdy_data.map(
            func=lambda x: "0" if x in ["D", "L", "P", "W"] else x
        )
        df_rdy_data = df_rdy_data.map(func=lambda x: "1" if x != "0" else x)
        rdy_data = df_rdy_data.to_numpy(dtype=int)
        return rdy_data

    def create_exits_in_matrix(self) -> None:
        """Opens every exit in the room's tiles, by writing void along its border.

        Exit spans are absolute skeleton coordinates, so both ends go through
        `_local_row`/`_local_col`. On the vertical axis that conversion flips,
        which reverses the span: `c1` is the *lower* edge of the opening and so
        the *higher* row index. Reading it as a forward range is what mirrored
        left/right openings, leaving two rooms sharing one opening with holes at
        different heights.
        """
        for side in ("left", "right", "up", "down"):
            for exit in self.exits[side]:
                c1, c2 = exit[0], exit[1]
                if side in ("left", "right"):
                    row_lo, row_hi = self._local_row(c2), self._local_row(c1)
                    column: int = 0 if side == "left" else -1
                    self.data[row_lo : row_hi + 1, column] = "0"
                else:
                    col_lo, col_hi = self._local_col(c1), self._local_col(c2)
                    row: int = 0 if side == "up" else -1
                    self.data[row, col_lo : col_hi + 1] = "0"

    # TODO: adapt this method so that it can also extract room path between
    # special points and exits - includes starting/ending room and strawberries
    def extract_room_path(self, max_iterations: int = 0) -> list[tuple[int, int]]:
        """Calls Celeste-A* to try to extract a path.

        Returns a path if found in the specified iterations, empty list otherwise.
        """
        # Preprocessing the room data
        room_astar_ready: np.ndarray[tuple[int, int], np.dtype[Any]] = (
            self.extract_astar_ready_data()
        )

        # Extracting exits ie. supposedly connected points
        room_exits_points: list[tuple[int, int]] = self.get_starting_ending_points()
        nb_exits: int = len(room_exits_points)

        if nb_exits < 2:
            return []  # avoid UnboundLocalError for uneligible rooms

        for k in range(nb_exits - 1):
            # use transitivity
            pt1, pt2 = room_exits_points[k], room_exits_points[k + 1]
            room_path: list[tuple[int, int]] = astar(
                room=self,
                maze=room_astar_ready,
                start=pt1,
                end=pt2,
                max_iterations=max_iterations,
            )
            if not room_path:
                return []

        # TODO: fix the output format - for now, if more than 2 exits, returns
        # the path of the last couple of exits only, not the others: still
        # okay for playability assertion, but not the cleanest solution.
        return room_path

    def is_playable_room(self, max_iterations: int = 0) -> bool:
        """Asserts whether a path has been found in the room or not."""
        room_path: list[tuple[int, int]] = self.extract_room_path(
            max_iterations=max_iterations
        )

        if not room_path:
            logger.warning(msg="A* room playability failed in current room.")
            return False

        logger.info(msg="Playability check succeeded - current room seems playable!")
        return True

    def is_valid_special(
        self, special_point: tuple[int, int], max_iter: int = 0
    ) -> bool:
        """Assert whether a special point is connected to all room exits."""
        room_astar_ready: np.ndarray[tuple[int, int], np.dtype[Any]] = (
            self.extract_astar_ready_data()
        )
        room_exits_points: list[tuple[int, int]] = self.get_starting_ending_points()

        for end_point in room_exits_points:
            sp_to_exits: list[tuple[int, int]] = astar(
                room=self,
                maze=room_astar_ready,
                start=special_point,
                end=end_point,
                max_iterations=max_iter,
            )
            if not sp_to_exits:
                logger.warning(msg="Special point cannot reach all room exits")
                return False

        logger.info(msg="Playability check succeeded - special point is valid")

        return True

    def add_respawn_points(self, clear_space_size: int = 2) -> None:
        """Adds a player entity just inside each entrance, so the room is enterable.

        The spawn sits a couple of tiles in from the opening it belongs to, so
        its position along that border comes from the same
        `_local_row`/`_local_col` conversion the opening itself was carved with.

        Args:
            clear_space_size: half-width of the pocket of void cleared around the
                spawn, so the player does not appear inside a wall.
        """
        for side in ("up", "down", "left", "right"):
            for exit in self.exits[side]:
                midpoint: int = int(np.mean(exit))
                row: int
                col: int
                if side == "up":
                    row, col = 2, self._local_col(midpoint)
                elif side == "down":
                    row, col = self.size[1] - 3, self._local_col(midpoint)
                elif side == "left":
                    row, col = self._local_row(midpoint), 2
                else:
                    row, col = self._local_row(midpoint), self.size[0] - 3

                self.data[row, col] = "P"
                self.data[row + 1, col - 1 : col + 2] = "D"
                for i, j in [
                    (a, b)
                    for a in range(-1 * clear_space_size + 1, clear_space_size)
                    for b in range(-1 * clear_space_size + 1, clear_space_size)
                    if (a, b) != (0, 0)
                ]:
                    # Clamped rather than wrapped: `_local_row` can put a spawn on
                    # row 0, where a negative index would silently clear tiles on
                    # the opposite wall instead of falling off the room.
                    if 0 <= row + i < self.size[1] and 0 <= col + j < self.size[0]:
                        self.data[row + i, col + j] = "0"

    def add_special_points(self, nb_tries_limit: int) -> bool:
        """Add structure with player spawn point or crystal heart depending on status.

        Returns a boolean whether it worked within nb_tries_limit tries.
        """
        if not self.status:
            logger.info(
                msg="This room is neither a starting not an ending room - skipping."
            )
            return True

        valid_sp = False
        try_iter = 0
        random_sp = (0, 0)
        while (
            not valid_sp and try_iter < nb_tries_limit
        ):  # avoids softlock in the middle of lvl gen
            try_iter += 1
            random_sp = tuple(
                np.random.randint([self.size[1] - 4, self.size[0] - 3])
                + np.array([3, 1])
            )
            valid_sp = self.is_valid_special(random_sp)

        if not valid_sp:  # something is probably wrong with room - need re-gen
            logger.warning(
                msg=f"No success in {nb_tries_limit} iterations - re-generating room."
            )
            return False

        logger.info(msg=f"Special point found in {try_iter} iterations.")

        if self.status == "end":
            self.data[random_sp[0], random_sp[1]] = "H"
        elif self.status == "start":
            self.data[random_sp[0], random_sp[1] - 1 : random_sp[1] + 2] = "D"
            self.data[random_sp[0] - 1, random_sp[1]] = "P"
        else:
            raise ValueError("Status should be either None, 'start' or 'end'!")
        return True

    def save(self, output_dir: str) -> None:
        """Exports all data contained in the room."""
        output_dir_path = Path(output_dir)
        json_room: RoomJSON = {
            "origin": list(self.origin),
            "size": list(self.size),
            "exits": self.exits,
        }

        np.savetxt(
            fname=output_dir_path / "data.csv",
            X=self.data,
            fmt="%s",
        )
        output_dir_path_metadata: Path = output_dir_path / "metadata.json"
        with output_dir_path_metadata.open(
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(obj=json_room, fp=f, ensure_ascii=False, indent=4)

        logger.info(msg=f"Saved lvl data in folder {output_dir}")

    def save_data(self, path_to_save: str) -> None:
        """Exports room.data in csv format."""
        np.savetxt(
            fname=f"{path_to_save}.csv",
            X=self.data,
            fmt="%s",
        )

        logger.info(msg=f"Saved lvl data in folder {path_to_save}")


class Celeskeleton:
    """Container for rooms + metadata."""

    def __init__(self) -> None:
        """Init a Celeskeleton object: container for rooms + metadata."""
        self.lvl: dict[str, Room] = {}
        self.starting_room: str | None = None
        self.ending_room: str | None = None

    def list_all_rooms(self) -> list[str]:
        """Lists all rooms in a given skeleton by name."""
        return list(self.lvl)

    def get_nb_rooms(self) -> int:
        """Returns number of rooms in a skeleton."""
        return len(list(self.lvl))

    def get_starting_room(self) -> str:
        """Returns starting room in a skeleton."""
        candidate_rooms: list[str] = [
            room for room in self.lvl if self.lvl[room].get_status() == "start"
        ]
        if len(candidate_rooms) > 1:
            raise ValueError(
                f"Multiple starting rooms have been found: {candidate_rooms}"
            )
        elif len(candidate_rooms) == 0:
            raise ValueError("No starting room have been found.")
        return candidate_rooms[0]

    def get_ending_room(self) -> str:
        """Returns ending room in a skeleton."""
        candidate_rooms: list[str] = [
            room for room in self.lvl if self.lvl[room].get_status() == "end"
        ]
        if len(candidate_rooms) > 1:
            raise ValueError(
                f"Multiple ending rooms have been found: {candidate_rooms}"
            )
        elif len(candidate_rooms) == 0:
            raise ValueError("No ending room have been found.")
        return candidate_rooms[0]

    def add_room(self, room: Room, room_name: str) -> None:
        """Adds a room to a skeleton (requires a name)."""
        if room_name in self.lvl:
            raise ValueError(
                f"Room {room_name} already exists! Please use another name."
            )
        self.lvl[room_name] = room

    def remove_room(self, room_name: str) -> None:
        """Removes an existing room from a skeleton (by name)."""
        if room_name in self.lvl:
            del self.lvl[room_name]
        else:
            raise ValueError(f"Room {room_name} does not exists!")

    def global_overlapping(self) -> list[tuple[str, str]]:
        """Returns tuples of all overlapping rooms in a skeleton."""
        nb_rooms: int = len(self.lvl)
        l_names_rooms: list[str] = list(self.lvl)
        l_index_overlaps: list[tuple[str, str]] = []
        for i in range(nb_rooms - 1):
            for j in range(i + 1, nb_rooms):
                if self.lvl[l_names_rooms[i]].is_overlapping(
                    room=self.lvl[l_names_rooms[j]]
                ):
                    l_index_overlaps.append((l_names_rooms[i], l_names_rooms[j]))
        return l_index_overlaps

    def is_overlapping_with(self, room: Room) -> bool:
        """Check if any room in the skeleton overlaps with a separated Room object."""
        return any(room.is_overlapping(room=self.lvl[name]) for name in self.lvl)

    def get_room_by_name(self, room_name: str) -> Room:
        """Returns room by name, will modify the original obj even if different name."""
        try:
            return self.lvl[room_name]
        except KeyError as err:
            raise err

    def select_last_room(self, room_name: str, p: float) -> Room:
        """Returns last room or random room in the skeleton with a proba p."""
        if np.random.rand() < p:
            name: str = np.random.choice(a=list(self.lvl))
        else:
            name = room_name
        return self.get_room_by_name(room_name=name)

    def set_start_end(self) -> None:
        """Sets starting_room and ending_room attributes of a skeleton.

        While at it, updates correspondant room status to match overall structure.
        """
        l_lvl_names: list[str] = list(self.lvl)
        self.starting_room, self.ending_room = l_lvl_names[0], l_lvl_names[-1]
        self.get_room_by_name(room_name=self.starting_room).set_status(status="start")
        self.get_room_by_name(room_name=self.ending_room).set_status(status="end")

    def get_special_rooms(self) -> dict[str, str]:
        """Maps the name of the starting/ending rooms to their status.

        A room is considered special either because its own status says so, or
        because the skeleton points at it through `starting_room`/`ending_room`
        (both are set by `set_start_end`, statuses are not).

        Returns:
            A `{room_name: "start" | "end"}` dict, empty if nothing is set yet.
        """
        special: dict[str, str] = {}
        if self.starting_room and self.starting_room in self.lvl:
            special[self.starting_room] = "start"
        if self.ending_room and self.ending_room in self.lvl:
            special[self.ending_room] = "end"
        for name, room in self.lvl.items():
            status: str | None = room.get_status()
            if status:
                special[name] = status

        return special

    def get_room_graph(self) -> dict[str, set[str]]:
        """Maps each room to the rooms it shares an opening with.

        `generate_celeskeleton` grows a level by attaching each new room to one
        existing room, so the graph this returns is a tree - n rooms and n-1
        connections - rooted at the first room added. Nothing here assumes that
        though: the graph is read back off the exits, so a hand-built or loaded
        skeleton is described just as faithfully.

        Note this is *not* the same relation as room numbering. Numbering is
        insertion order, and with `generate_celeskeleton`'s `p` above 0 a new
        room attaches to a random earlier one, so consecutively numbered rooms
        are frequently not neighbours at all.

        Returns:
            A `{room_name: {neighbour_name, ...}}` adjacency map, with an entry
            for every room including the ones nothing connects to.
        """
        names: list[str] = list(self.lvl)
        graph: dict[str, set[str]] = {name: set() for name in names}
        for i in range(len(names) - 1):
            for j in range(i + 1, len(names)):
                name_i, name_j = names[i], names[j]
                if self.lvl[name_i].is_connected_to(room=self.lvl[name_j]):
                    graph[name_i].add(name_j)
                    graph[name_j].add(name_i)

        return graph

    def get_traversal_path(self) -> list[str]:
        """Returns the rooms the player has to cross, start to end.

        This is the level's actual traversal order, and the thing room numbering
        is mistaken for: on a tree there is exactly one route between two rooms,
        so the rooms on it are the ones a run must pass through and every other
        room is an optional branch off it.

        Returns:
            Room names from the starting room to the ending room, both included.

        Raises:
            ValueError: no start or end room is set, or nothing connects the two.
        """
        start: str = self.get_starting_room()
        end: str = self.get_ending_room()
        graph: dict[str, set[str]] = self.get_room_graph()

        # Breadth-first from the start, remembering how each room was reached.
        # The graph is a tree by construction, so the first route found is the
        # only one; `sorted` keeps the walk deterministic if it ever is not.
        came_from: dict[str, str | None] = {start: None}
        queue: deque[str] = deque([start])
        while queue:
            current: str = queue.popleft()
            if current == end:
                path: list[str] = []
                room_name: str | None = current
                while room_name is not None:
                    path.append(room_name)
                    room_name = came_from[room_name]
                return path[::-1]
            for neighbour in sorted(graph[current]):
                if neighbour not in came_from:
                    came_from[neighbour] = current
                    queue.append(neighbour)

        raise ValueError(
            f"No path connects {start!r} to {end!r}: the skeleton is not "
            "traversable, so no level generated from it can be finished."
        )

    def show_skeleton(
        self,
        show_names: bool = True,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> Axes:
        """Plots the skeleton as a 2D map of rooms.

        Each room is a rectangle, exits are drawn as thick red segments on the
        room borders, and the starting/ending rooms - if any - are filled with
        their own color.

        Args:
            show_names: writes the name of each room at its center.
            save_path: file the figure is written to, format from the extension
                (`.png`, `.svg`, ...). Nothing is saved when left to None.
            show: opens the figure in a window, a no-op on headless backends.
                Notebooks display the returned axes on their own, no need for it.

        Returns:
            The axes the skeleton has been drawn on, to tweak it further.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        special: dict[str, str] = self.get_special_rooms()

        for name, room in self.lvl.items():
            x, y = room.get_origin()
            w, h = room.get_size()
            status: str | None = special.get(name)
            ax.add_patch(
                p=Rectangle(
                    xy=(x, y),
                    width=w,
                    height=h,
                    facecolor=to_rgba(c=STATUS_COLORS[status], alpha=0.25)
                    if status
                    else "none",
                    edgecolor=ROOM_EDGE_COLOR,
                    linewidth=1.0,
                )
            )
            for x0, y0, x1, y1 in room.exits_to_list():
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=EXIT_COLOR,
                    linewidth=3.0,
                    solid_capstyle="butt",
                )
            if show_names:
                ax.annotate(
                    text=name,
                    xy=(x + w / 2, y + h / 2),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=ROOM_EDGE_COLOR,
                )

        self._set_skeleton_plot_limits(ax=ax)
        ax.set_aspect(aspect="equal")
        ax.set_axisbelow(b=True)
        ax.grid(visible=True, linewidth=0.3, alpha=0.4)
        ax.set_title(label=f"Celeskeleton - {self.get_nb_rooms()} rooms")
        ax.legend(
            handles=[
                Patch(
                    facecolor=to_rgba(c=STATUS_COLORS[status], alpha=0.25),
                    edgecolor=ROOM_EDGE_COLOR,
                    label=f"{status} room",
                )
                for status in ("start", "end")
                if status in special.values()
            ]
            + [
                Line2D(
                    xdata=[], ydata=[], color=EXIT_COLOR, linewidth=3.0, label="exit"
                )
            ],
            loc="best",
            fontsize=8,
        )

        if save_path:
            fig.savefig(fname=save_path, dpi=150, bbox_inches="tight")
            logger.info(msg=f"Saved skeleton plot to {save_path}.")
        if show:
            plt.show()

        return ax

    def _set_skeleton_plot_limits(self, ax: Axes, margin_ratio: float = 0.05) -> None:
        """Fits the axes around every room, with a small margin around them."""
        corners: list[list[int]] = [room.get_corners() for room in self.lvl.values()]
        if not corners:
            logger.warning(msg="The skeleton object is empty - nothing to plot!")
            return

        xs: list[int] = [c[0] for c in corners] + [c[2] for c in corners]
        ys: list[int] = [c[1] for c in corners] + [c[3] for c in corners]
        margin: float = margin_ratio * max(max(xs) - min(xs), max(ys) - min(ys), 1)
        ax.set_xlim(left=min(xs) - margin, right=max(xs) + margin)
        ax.set_ylim(bottom=min(ys) - margin, top=max(ys) + margin)

    def save(self, output_dir: str) -> None:
        """Exports all data contained in the skeleton in a readable JSON."""
        json_skeleton: SkeletonJSON = {}

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        for room in self.lvl:
            json_skeleton[room] = {
                "origin": list(self.lvl[room].origin),
                "size": list(self.lvl[room].size),
                "exits": self.lvl[room].exits,
            }
            np.savetxt(
                fname=output_dir_path / f"{room}.csv",
                X=self.lvl[room].data,
                fmt="%s",
            )

        output_dir_path_data: Path = output_dir_path / "data.json"
        with output_dir_path_data.open(
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(obj=json_skeleton, fp=f, ensure_ascii=False, indent=4)

        logger.info(msg=f"Saved lvl data in folder {output_dir}.")

    def format_filled_celeskeleton(self) -> None:
        """Format each room of the skeleton by adding borders.

        To refactor with the generation algorithm.
        """

        def _create_full_borders_room(data: np.ndarray, tile_id: int = 1) -> np.ndarray:
            data[0, :] = tile_id
            data[-1, :] = tile_id
            data[:, 0] = tile_id
            data[:, -1] = tile_id

            return data

        for room_name in self.list_all_rooms():
            room: Room = self.get_room_by_name(room_name)
            room.set_data(data=_create_full_borders_room(room.data))
            room.create_exits_in_matrix()

    def extract_rooms_metadata(
        self,
    ) -> list[tuple[tuple[int, int], tuple[int, int], dict[str, list[list[int]]]]]:
        """Return metadata list for each room of a skeleton: size, origin, exits."""
        rooms_metadata: list[
            tuple[tuple[int, int], tuple[int, int], dict[str, list[list[int]]]]
        ] = []
        for room_name in self.list_all_rooms():
            room: Room = self.get_room_by_name(room_name)
            rooms_metadata.append((room.size, room.origin, room.exits))

        return rooms_metadata


def generate_celeskeleton(
    nb_rooms: int,
    p: float = 0.5,
    room_size: tuple[int, int] | None = None,
) -> Celeskeleton:
    """Generates procedurally a level skeleton.

    Skeleton made of nb_rooms rooms
    Probability p of connecting the new room for each iteration,
    instead of taking the last existing room:
    - p = 0 generates a pathway style skeleton
    - p > 0 generates a labyrinth style skeleton (p close to 0 -> pathway-like)

    Returns a fully functionnal skeleton.
    """
    skeleton = Celeskeleton()

    if room_size:
        size_init: tuple[int, int] = room_size
    else:
        w, h = random.randint(a=10, b=100), random.randint(a=10, b=100)
        size_init = (w, h)

    room_init = Room(width=size_init[0], height=size_init[1])
    sides: list[str] = ["up", "down", "left", "right"]

    k = 1
    room_name = f"room_{k}"

    skeleton.add_room(room_init, room_name)

    while skeleton.get_nb_rooms() < nb_rooms:
        k += 1
        overlapping: bool = True
        while overlapping:
            last_room: Room = skeleton.select_last_room(room_name, p)
            side: str = np.random.choice(a=sides)
            if room_size:
                room = Room(width=room_size[0], height=room_size[1])
            else:
                room = Room(
                    width=random.randint(a=10, b=100),
                    height=random.randint(a=10, b=100),
                )
            try:
                room.connect(last_room, side, skeleton)
            except ValueError:
                continue
            overlapping = skeleton.is_overlapping_with(room)
        room_name = f"room_{k}"
        skeleton.add_room(room, room_name)

    skeleton.set_start_end()
    starting_room: str = skeleton.get_starting_room()
    ending_room: str = skeleton.get_ending_room()
    skeleton.get_room_by_name(room_name=starting_room).set_status(status="start")
    skeleton.get_room_by_name(room_name=ending_room).set_status(status="end")

    return skeleton


def json_to_skeleton(data_json: SkeletonJSON) -> Celeskeleton:
    """Rebuilds a Celeskeleton from an already parsed `data.json` payload.

    Args:
        data_json: room name -> room payload, see `RoomJSON`.

    Returns:
        The reconstructed skeleton, with start/end rooms set.
    """
    json_skeleton = Celeskeleton()
    for name, room_json in data_json.items():
        w, h = room_json["size"]
        x, y = room_json["origin"]
        room = Room(width=w, height=h, origin_x=x, origin_y=y)
        room.set_exits(room_json["exits"])
        json_skeleton.add_room(room, name)

    json_skeleton.set_start_end()

    return json_skeleton


def load_data_to_celeskeleton(folder_path: str | Path) -> Celeskeleton:
    """Reads a skeleton folder (`data.json` + one csv per room) back into memory.

    Args:
        folder_path: directory previously written by `Celeskeleton.save`.

    Returns:
        The skeleton, with each room's data matrix restored from its csv.
    """
    folder = Path(folder_path)
    with (folder / "data.json").open(encoding="utf-8") as f:
        # json.load is Any: this annotation is the trust boundary where the
        # payload is declared to follow the SkeletonJSON schema.
        json_file: SkeletonJSON = json.load(fp=f)
    skeleton: Celeskeleton = json_to_skeleton(data_json=json_file)
    for i in range(len(skeleton.list_all_rooms())):
        temp_data: np.ndarray[tuple[int, int], np.dtype[Any]] = pd.read_csv(
            filepath_or_buffer=folder / f"room_{i + 1}.csv",
            header=None,
            sep=" ",
            dtype=str,
        ).to_numpy()
        skeleton.get_room_by_name(room_name=f"room_{i + 1}").set_data(temp_data)

    return skeleton
