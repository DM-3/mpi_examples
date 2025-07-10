import numpy as np
import cv2 as cv
import time
import argparse


def read_parameters() -> tuple[int, int, float, bool]:
    """
    Parse command line arguments for Game of Life simulation.

    Returns:
    -------
    N : int
        Problem size (grid dimension)

    t_steps : int
        Number of timesteps

    probability : float
        Initial probability of alive cells

    show_flag : bool
        Whether to display the board
    """
    parser = argparse.ArgumentParser(
        description='Game of Life Simulation - MPI Version',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--N',
        type=int,
        help='Problem size (grid dimension)',
        default=100,
        required=False
    )
    parser.add_argument(
        '--t_steps',
        type=int,
        help='Number of timesteps',
        default=100,
        required=False
    )
    parser.add_argument(
        '--probability',
        type=float,
        help='Initial probability of alive cells',
        default=0.4,
        required=False
    )
    parser.add_argument(
        '--show',
        action="store_true",
        help='Show mesh during simulation',
        default=False,
        required=False
    )

    try:
        args = parser.parse_args()

        if args.N <= 0:
            raise ValueError("Problem size must be positive")
        if args.t_steps < 0:
            raise ValueError("Number of timesteps must be non-negative")
        if not 0.0 <= args.probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1")

        return args.N, args.t_steps, args.probability, args.show

    except (ValueError, SystemExit):
        print("Usage: python game_of_life.py --N <N> --t_steps <t_steps> "
              "--probability <probability> [--show]")
        print("Using default values: N=100, t_steps=100, "
              "probability=0.4, show=True")
        return 100, 100, 0.4, True


def create_random_board(size: tuple[int, int], probability: float) -> np.ndarray[np.uint8]:
    """
    Initialize the board randomly with given probability of alive cells.

    Args:
    -----
    size : tuple[int, int]
        board height and width

    probability : float
        Probability of a cell being alive initially
    """
    np.random.seed(0)
    return np.where(np.random.rand(*size) > probability, np.uint8(1), np.uint8(0))


def update_step(source: np.ndarray[np.uint8], target: np.ndarray[np.uint8]):
    
    # copy torus padding
    source[0,:]  = source[-2,:]
    source[-1,:] = source[1,:]
    source[:,0]  = source[:,-2]
    source[:,-1] = source[:,1]

    source[0,0]   = source[-2,-2]
    source[0,-1]  = source[-2,1]
    source[-1,0]  = source[1,-2]
    source[-1,-1] = source[1,1]

    # count neighbors for all pixels
    w = np.asarray([[1,1,1],
                    [1,0,1],
                    [1,1,1]], dtype=np.uint8)
    count = cv.filter2D(source, -1, w)

    # apply rules
    target &= 0
    target |= np.where(source + count == 3, np.uint8(1), np.uint8(0))   # alive cells with 2 neighbors become alive
    target |= np.where(count == 3, np.uint8(1), np.uint8(0))            # any cell with 3 neighbors becomes alive


def main():
    # get problem parameters
    N, t_steps, probability, show = read_parameters()

    print(f"Game of Life with N = {N}, {t_steps} timesteps "
          f"and initial probability = {probability}")

    # initialize board and temporary board for computations
    padding = 2
    WIDTH = N + padding
    HEIGHT = N + padding
    board = create_random_board((HEIGHT, WIDTH), probability)
    t_board = np.zeros_like(board)

    # update loop
    if show:
        while cv.waitKey(1) != 27:
            cv.imshow('board', cv.resize(board[1:-1,1:-1] * 255, 
                                (900, 900), 
                                interpolation=cv.INTER_NEAREST))
            update_step(board, t_board)         # update board
            board, t_board = t_board, board     # swap with temporary board
    else:
        start = time.time()

        for _ in range(t_steps):
            update_step(board, t_board)         # update board
            board, t_board = t_board, board     # swap with temporary board

        end = time.time()
        print(f'computation took {end-start:.6f} seconds')


if __name__ == "__main__":
    main()
