from mpi4py import MPI
import numpy as np
import sortednp
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# guarantee that the number of ranks is a power of 2
if not ((n_ranks & (n_ranks - 1) == 0) and n_ranks != 0):
    sys.exit("please use a number of ranks that is a power of 2")

n_elems = 10000000
if (n_elems % n_ranks) != 0:
    n_elems += n_ranks - (n_elems % n_ranks)
n_elems_per_rank = n_elems // n_ranks


# generate random array and broadcast to all ranks
array = np.random.rand(n_elems) if rank == 0 else None

# allocate receive buffer
section = np.empty(n_elems_per_rank, dtype=np.float64)

# scatter data array across ranks
comm.Scatter([array, MPI.DOUBLE], [section, MPI.DOUBLE])


# sort local section
section = np.sort(section)


# gradually merge
i = 2    
while i <= n_ranks:

    if (rank % i) == i//2:
        comm.send(section, dest = rank - i//2, tag=100)

    if (rank % i) == 0:
        other_section = comm.recv(source = rank + i//2, tag=100)
        section = sortednp.merge(section, other_section)

    i <<= 1


# final array on rank 0
if rank == 0:
    array = section
    print(f'is sorted: {all(array[i] <= array[i+1] for i in range(len(array) - 1))}')
