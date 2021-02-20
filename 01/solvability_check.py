from math import sqrt

import npuzzle


def is_solvable(env):
    '''
    True or False?
    Tady naprogramujte svoje reseni
    '''
    # noinspection PyProtectedMember
    tiles = env._NPuzzle__tiles  # nasty little hack to get all tiles directly from the object
    clean_tiles = [tile for tile in tiles if tile is not None]
    total_inversions = 0
    for idx, element in enumerate(clean_tiles):
        inversions = [tested for tested in clean_tiles[idx + 1:] if tested < element]
        total_inversions += len(inversions)

    size = int(sqrt(len(tiles)))
    if size % 2 == 1:
        # If the width is odd, then every solvable state
        # has an even number of inversions.
        return total_inversions % 2 == 0
    else:
        none_tile_pos = tiles.index(None)  # find idx (from 0) of empty tile
        row_idx = int(none_tile_pos / size)  # find idx (from 0) of the row
        row_from_bottom = size - row_idx  # number row (starting from 1) from bottom

        if row_from_bottom % 2 == 1:
            # an even number of inversions if the blank is
            # on an odd numbered row counting from the bottom
            return total_inversions % 2 == 0
        else:
            # an odd number of inversions if the blank is on an even
            # numbered row counting from the bottom;
            return total_inversions % 2 == 1


if __name__ == "__main__":
    env = npuzzle.NPuzzle(4)
    env.reset()
    env.visualise()
    # just check
    print(is_solvable(env))
