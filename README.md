# MPI Examples
Examples for parallizing code snippets with mpi (Message Passing Interface).

- [Game of Life](#game-of-life)
- [Merge Sort](#merge-sort)


## Game of Life
Distributes the board and update computation for Conway's game of life across multiple ranks.

running:
```
mpiexec -n 2 python3 game_of_life.py --N 1000 --t_steps 1000 --show --probability 0.3
```


## Merge Sort
Each rank sorts a section of an input array individually, then merges them back together.

running:
```
mpiexec -n 2 python3 merge_sort.py
```
