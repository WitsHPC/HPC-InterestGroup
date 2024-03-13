#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>


int neighbours(int x, int y, int W, int H, int* cells) {
    int N = 0;
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
                int idx = ty * W + tx;
                N += cells[idx];
            }
        }
    }
    return N;
}

void print(int W, int H, int* cells) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (cells[y * W + x])
                printf("#");
            else 
                printf(".");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {

    int w, h, n, m, A, B, C;
    int _ = scanf("%d %d %d %d %d %d %d\n", &w, &h, &n, &m, &A, &B, &C);
    int* cells = (int*) malloc(w * h * sizeof(int));
    int* buffer = (int*) malloc(w * h * sizeof(int));
    char c;
    // now read the grid
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            int _ = scanf("%c", &c);
            if (c == '#') 
                cells[i * w + j] = 1;
            else
                cells[i * w + j] = 0;
        }
        int _ = scanf("%c", &c); // newlines
    }

    int i = 0;
    float angle = 0;
    // Used to get a consistent frame rate
    for (int step=0; step < n; ++step){
        int step_one_indexed = step + 1;
        // update the new array
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int N = neighbours(x, y, w, h, cells);
                int idx = y * w + x;
                int is_on = cells[idx];
                int new_val = 0;
                if (is_on){
                    if (N < A || N > B) new_val = 0;
                    else new_val = 1;
                }else{
                    if (N == C) new_val = 1;
                }
                buffer[idx] = new_val;
            }
        }
        // Swap the vectors so that the updated one is drawn in the next frame.
        int * temp = buffer;
        buffer = cells;
        cells = temp;

        if (step == 0 || (step + 1) % m == 0 || step == n - 1) 
        {
            print(w, h, cells);
        }
    }
    free(cells);
    free(buffer);
}