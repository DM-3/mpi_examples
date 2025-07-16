# usage example:
# mpiexec -n 4 python3 game_of_life.py --N 1000 --t_steps 1000 --show --probability 0.3

from mpi4py import MPI
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


def create_random_board(
        size: tuple[int, int], 
        probability: float, 
        seed: int) -> np.ndarray[np.uint8]:
    """
    Initialize the board randomly with given probability of alive cells.

    Args:
    -----
    size : tuple[int, int]
        board height and width

    probability : float
        probability of a cell being alive initially
    
    seed : int
        seed for random number generation
    """
    np.random.seed(seed)
    return np.where(np.random.rand(*size) < probability, np.uint8(1), np.uint8(0))


def update_step(
        source: np.ndarray[np.uint8], 
        target: np.ndarray[np.uint8],
        comm,
        rank: int,
        n_ranks: int):
    
    # copy local torus padding columns
    source[:,0] = source[:,-2]
    source[:,-1] = source[:,1]

    # communicate global torus padding rows
    # cycle up in rank
    comm.Barrier()
    dst = (rank + 1 + n_ranks) % n_ranks
    src = (rank - 1 + n_ranks) % n_ranks
    source[0,:] = comm.sendrecv(source[-2,:], dest=dst, source=src)
    # cycle down in rank
    comm.Barrier()
    dst = (rank - 1 + n_ranks) % n_ranks
    src = (rank + 1 + n_ranks) % n_ranks
    source[-1,:] = comm.sendrecv(source[1,:], dest=dst, source=src)

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

    # initialize mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # initialize board and temporary board for computations
    padding = 2
    WIDTH = N + padding
    HEIGHT = N // n_ranks + padding
    board = create_random_board((HEIGHT, WIDTH), probability, rank)
    t_board = np.zeros_like(board)

    # update loop
    if show:
        while cv.waitKey(1) != 27:
            cv.imshow(f'{rank} board', cv.resize(board[1:-1,1:-1] * 255, 
                                (900, 900 // n_ranks), 
                                interpolation=cv.INTER_NEAREST))
            update_step(board, t_board, comm, rank, n_ranks)    # update board
            board, t_board = t_board, board                     # swap with temporary board
    else:
        start = time.time()

        for _ in range(t_steps):
            update_step(board, t_board, comm, rank, n_ranks)    # update board
            board, t_board = t_board, board                     # swap with temporary board

        end = time.time()
        print(f'computation took {end-start:.6f} seconds')


if __name__ == "__main__":
    main()
