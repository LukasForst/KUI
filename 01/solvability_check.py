import npuzzle


def is_solvable(env):
    '''
    True or False?
    Tady naprogramujte svoje reseni
    '''


if __name__ == "__main__":
    env = npuzzle.NPuzzle(3)
    env.reset()
    env.visualise()
    # just check
    print(is_solvable(env))
