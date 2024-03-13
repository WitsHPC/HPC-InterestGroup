#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

int neighbours(int x, int y, int W, int H, bool* cells) {
    int N = 0;
    int tx, ty;
    // for all neighbours
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // not the cell itself
            if (!i && !j) continue;
            tx = x + i;
            ty = y + j;

            // // wrap
            if (tx == -1 || tx == W)
                tx = (tx + W) % W;
            if (ty == -1 || ty == H)
                ty = (ty + H) % H;

            // add if the cell is active
            int idx = ty * W + tx;
            N += cells[idx];
        }
    }
    return N;
}

void print(int W, int H, bool* cells) {
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
    omp_set_num_threads(4);
    setvbuf(stdout, NULL, _IOFBF, 16384 * 16); // https://stackoverflow.com/a/65020999

    int w, h, n, m, A, B, C;
    int _ = scanf("%d %d %d %d %d %d %d\n", &w, &h, &n, &m, &A, &B, &C);
    bool* cells = (bool*) malloc(w * h * sizeof(bool));
    bool* buffer = (bool*) malloc(w * h * sizeof(bool));
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
    for (int step=0; step < n; ++step){
        // update the new array
        #pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int N = neighbours(x, y, w, h, cells);
                int idx = y * w + x;
                bool is_on = cells[idx];
                bool new_val = 0;
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
        bool * temp = buffer;
        buffer = cells;
        cells = temp;


        // print when we need to
        if (step == 0 || (step + 1) % m == 0 || step == n - 1) 
        {
            print(w, h, cells);
        }
    }
    free(cells);
    free(buffer);
}