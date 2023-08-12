# Problem
## Game of Life
Conway's Game of Life is a cellular automaton that is played on a grid of cells. Each cell can be either alive or dead. The game progresses in discrete steps. At each step, the following rules are applied to each cell:
1. Any live cell with fewer than two live neighbours dies, as if by underpopulation.
2. Any live cell with two or three live neighbours lives on to the next generation.
3. Any live cell with more than three live neighbours dies, as if by overpopulation.
4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
5. A cell's neighbours are those cells which are horizontally, vertically or diagonally adjacent. Most cells will have eight neighbours. We also follow wrapping conventions, so that the top row is adjacent to the bottom row, and the left column is adjacent to the right column.

Furthermore, we do not update the grid in place. This means we first compute the next state of the grid using the current state and then update the grid with the new state. Notably, this is often done using two arrays, one for the current state and one for the next state. This is because the next state of a cell depends on the current state of the cell and its neighbours. If we were to update the grid in place, then we would be using the new state of a cell to compute the new state of its neighbours, which is not what we want.

### Our parametrisation
For this challenge, we make a small change to the rules above. Instead of having a fixed number of neighbours for each of the rules, we allow the number of neighbours to be a parameter. In particular, we have three parameters `A`, `B`, and `C` and the rules are as follows:
1. Any live cell with fewer than `A` live neighbours dies, as if by underpopulation.
2. Any live cell with between `A` or `B` live neighbours lives on to the next generation (inclusive).
3. Any live cell with more than `B` live neighbours dies, as if by overpopulation.
4. Any dead cell with exactly `C` live neighbours becomes a live cell, as if by reproduction.
## Input
Your input will be passed in via standard in and will consist of several lines. The first line will be:


`w h n m A B C`, where:
- `w` is the width of the grid
- `h` is the height of the grid
- `n` is the number of steps to simulate
- `m` is the *output frequency*, i.e. the number of steps between each output
- Then, `A`, `B`, and `C`, indicate the numbers for the rules mentioned above

We guarantee that the parameters will have the following bounds:
$$
10 \leq w, h, \leq 10^4 \\
0 \leq A, B, C \leq 8 \\
1 \leq n \leq 10^6 \\
1 \leq m \leq 10^8 \\
$$

The next `w` lines will indicate the initial state. Each line will have `h` characters, each of which will be either `.` (dead) or `#` (alive).

## Output

The output must be the state of the grid, every `m` steps, for `n` steps. The output must be in the same format as the input.
For instance, if `n = 1001` and `m = 250`, we output the following:
- The result after the first iteration
- The result after the 251st iteration
- The result after the 501st iteration
- The result after the 751st iteration
- The result after the 1001st iteration


You must always output the final state of the grid, but do not output it twice. Furthermore, the state after the first iteration must also always be printed, regardless of the value of `m`.

## Rules
- You can use Python, C or C++. You are allowed to use OpenMP, MPI or CUDA but you do not have to use any parallelisation library; you are allowed to write it in vanilla C/C++.
- The code must be your own. You are not allowed to:
  - Google \<How to make game of life fast\> and copy paste the code.
  - Get code from other teams.
  - Use a generative model to write code for you.
- You are allowed to use the internet to debug your code, but you must be able to explain every line of code you have written. 
- The resources linked to in the CUDA talk are fair game when it comes to optimisation.

## Submission Format
You must submit a zip file containing everything needed to run your code. In particular, it must have the following:
- A `build.sh` script that I'll run as `./build.sh`. This should compile your code
- A `run.sh` script that I'll run as `./run.sh`. This should run your code and print the output.
- A README file explaining the approach you followed and what optimisations you made.
- In the README file, include your team name and the name of your team members.
- We will time your code on a set of unknown test cases, and the code that is the fastest wins. Expect a balance where `m=1`, i.e., every state must be printed and where `m > n`, i.e. just the first and final states are printed.
- Submit on Moodle, link [here](https://courses.ms.wits.ac.za/moodle/mod/assign/view.php?id=21473).

You are allowed to submit multiple times, but to prevent spam we will limit this to 15 submissions per team.

## Examples
### Example 1
Input
```
10 10 1000000 100000000 2 3 3
..........
.#........
.#........
..........
.....#..#.
.......##.
........#.
#.........
..........
..........
```

Output
```
..........
..........
..........
..........
.......##.
.......###
.......###
..........
..........
..........
..........
..........
..........
..........
.......#..
......#.#.
......#..#
.......##.
..........
..........
```

Explanation:
The map is of size `10x10`, the number of iterations to perform is
`1000000` and you should print every `100000000` steps. This amounts to printing the first iteration and the last one.

Next, we have `A=2`, `B=3`, `C=3`, which define the rules.


### Example 2
See the file `2.in` for the input and `2.out` for the output.


## Suggestions
We strongly suggest you use input/output redirection to test your code. Particularly, run something like:

```bash
time ./mycode < examples/1.in > 1.out.mine
```

This will make the standard input come from the file `examples/1.in` and it will redirect the output to the file `1.out.mine`. 


Similarly, for the second example
```bash
time ./mycode < examples/2.in > 2.out.mine
```
The `time` command should tell you how long the code took to run. 
Then, to check correctness, run `diff examples/1.out 1.out.mine`. If nothing gets printed, then your code is correct. If something gets printed, then your code is incorrect.

**Important** Do not output redirect to the output file containing the correct answer, as this will overwrite it. Instead, redirect to a file with a different name, e.g. `1.out.mine`. If you have done this, just redownload the file from Github.




