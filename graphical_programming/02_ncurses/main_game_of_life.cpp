#include <curses.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#define ITERMAX 10000

/**
 * @brief Returns the current number of microseconds since epoch
 * 
 * @return int 
 */
int current_microseconds(){
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() * 1000;
}

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

int main(int argc, char** argv) {
    std::string character = "#";
    int speed = 60;

    if (argc >= 2){
        character = argv[1];
        if (character == "--help"){
            printf("Usage ./main [CHAR] [SPEED] \n");
            printf("\tExamples: ./main \t-> uses '#' and 60 FPS\n");
            printf("\tExamples: ./main . \t-> uses '.' and 60 FPS\n");
            printf("\tExamples: ./main . 2 \t-> uses '.' and 2 FPS\n");
            printf("\tExamples: ./main ■ 120 \t-> uses '■' and 120 FPS\n");
            return 0;
        }
    }
    if (argc >= 3){
        speed = atoi(argv[2]);
    }

    // need to set this locale to make some unicode characters work
    setlocale(LC_ALL, "");
    
    // Init
    cbreak();
    initscr();
    noecho();
    clear();
    
    std::vector<std::vector<int>> cells(LINES, std::vector<int>(COLS));
    std::vector<std::vector<int>> buffer(LINES, std::vector<int>(COLS));

    // set up randomly
    for (auto &row : cells) {
        for (auto &i : row) {
            i = rand() % 2;
        }
    }

    int i = 0;
    float angle = 0;
    // Used to get a consistent frame rate
    int time_now = current_microseconds();
    while (1) {
        erase();
        // Draw
        for (int y = 0; y < cells.size(); ++y) {
            for (int x = 0; x < cells[y].size(); ++x) {
                if (cells[y][x])
                    mvaddstr(y, x, character.c_str());
            }
        }
        refresh();
        // update the new array
        for (int y = 0; y < cells.size(); ++y) {
            for (int x = 0; x < cells[y].size(); ++x) {
                int N = neighbours(x, y, cells);
                int is_on = cells[y][x];
                int new_val = 0;
                if (is_on){
                    if (N < 2 || N > 3) new_val = 0;
                    else new_val = 1;
                }else{
                    if (N == 3) new_val = 1;
                }

                buffer[y][x] = new_val;
            }
        }
        // Swap the vectors so that the updated one is drawn in the next frame.
        std::swap(buffer, cells);
        int new_time = current_microseconds();
        // want to sleep 1000 / speed milliseconds
        int time_to_sleep = std::max(1e6 / speed - (new_time - time_now), 0.0);
        usleep(time_to_sleep);
        time_now = current_microseconds();
    }
    endwin();
    exit(0);
}
