# sliding-tile-solver

## About
This is a sliding-tile puzzle solver using IDA\*.  A sliding tile puzzle
with X tiles has X!/2 possible combinations.

### 4-tile solver
4!/2 is 12, all combinations are stored in lookup-table-4.txt

### 9-tile solver
9!/2 is 181440, all combinations are stored in lookup-table-9.txt

### 16-tile solver
16!/2 is 10,461,394,944,000 which is too large for me to build so I built part
of it and use IDA\* to find a sequence of moves that puts the puzzle in a state
that is in in the partial table I built.

My partial table consist of all 16-tile states that can be reached within
24-moves which is 42,928,799 combinations. This is stored in lookup-table-16.txt
which will be downloaded from https://github.com/dwalton76/sliding-tile-solver-lookup-tables
the first time you run the solver.

Three heuristic tables will also be downloaded
- lookup-table-16-x-1-6.txt
- lookup-table-16-x-7-12.txt
- lookup-table-16-x-13-15.txt

These are used by IDA\* to prune branches that are headed in the wrong direction.


## Examples
### 4-tile example
```
./solver.py --solve 3,0,2,1
```

### 9-tile example
```
./solver.py --solve 8,7,6,5,4,1,3,0,2
```

### 16-tile example
```
./solver.py --solve 9,11,12,15,7,2,13,8,1,0,5,4,10,6,14,3
```

### 25-tile example
```
./solver.py --solve 8,13,10,18,9,14,7,17,4,0,12,3,2,1,5,16,19,20,11,24,21,23,6,15,22
```

### 36-tile example
```
./solver.py --solve 1,8,3,17,23,12,19,7,11,29,5,6,0,9,2,4,10,16,26,31,34,32,22,15,20,13,30,14,21,24,25,27,33,28,18,35
```
