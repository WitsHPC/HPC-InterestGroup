#include <curses.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * @brief How many neighbours does a specific cell have
 * 
 * @param x 
 * @param y 
 * @param cells 
 * @return int 
 */
int neighbours(int x, int y, std::vector<std::vector<int>> &cells) {
    int N = 0;
    int W = cells[0].size();
    int H = cells.size();
    // for all neighbours
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // not the cell itself
            if (!i && !j) continue;
            int tx = x + i;
            int ty = y + j;
            // wrap
            tx = (tx + W) % W;
            ty = (ty + H) % H;
            // add if the cell is active
            if (tx >= 0 && tx < W && ty >= 0 && ty < H) {
                N += cells[ty][tx];
            }
        }
    }
    return N;
}

void print(std::vector<std::vector<int>> &cells) {
    string s = "";
    for (int y = 0; y < cells.size(); ++y) {
        for (int x = 0; x < cells[y].size(); ++x) {
            if (cells[y][x])
                s += "#";
            else 
                s += ".";
        }
        s += "\n";
    }
    cout << s;
}

int main(int argc, char** argv) {

    int w, h, n, m, A, B, C;
    cin >> w >> h >> n >> m >> A >> B >> C;
    std::vector<std::vector<int>> cells(h, std::vector<int>(w, 0));
    // now read the grid
    for (int i = 0; i < h; ++i) {
        string s;
        cin >> s;
        for (int j = 0; j < w; ++j) {
            if (s[j] == '#') {
                cells[i][j] = 1;
            }
        }
    }

    std::vector<std::vector<int>> buffer = cells;

    int i = 0;
    float angle = 0;
    // Used to get a consistent frame rate
    for (int step=0; step < n; ++step){
        int step_one_indexed = step + 1;

        // update the new array
        for (int y = 0; y < cells.size(); ++y) {
            for (int x = 0; x < cells[y].size(); ++x) {
                int N = neighbours(x, y, cells);
                int is_on = cells[y][x];
                int new_val = 0;
                if (is_on){
                    if (N < A || N > B) new_val = 0;
                    else new_val = 1;
                }else{
                    if (N == C) new_val = 1;
                }
                buffer[y][x] = new_val;
            }
        }
        // Swap the vectors so that the updated one is drawn in the next frame.
        std::swap(buffer, cells);

        if (step == 0 || (step + 1) % m == 0 || step == n - 1) 
        {
            print(cells);
        }
    }
}