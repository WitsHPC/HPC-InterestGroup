#include <stdio.h>

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

typedef std::vector<std::vector<bool>> Cells;

/**
 * @brief Calculates the number of live neighbours that this cell has.
 * 
 * @param cells 
 * @param row 
 * @param col 
 * @return int 
 */
int num_neighbours(const Cells& cells, int row, int col) {
    int w = cells[0].size();
    int h = cells.size();

    int sum_alive = 0;
    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            if (i == j && i == 0) continue;
            // wraparound effect.
            int temp_col = (col + i + w) % w;
            int temp_row = (row + j + h) % h;
            sum_alive += cells[temp_row][temp_col];
        }
    }
    return sum_alive;
};

/**
 * @brief This performs the logic to determine if the current cell needs to change to being alive / dead.
 * 
 * @param cells 
 * @param row 
 * @param col 
 * @return true 
 * @return false 
 */
bool get_new_state(const Cells& cells, int row, int col) {
    bool is_alive = cells[row][col];
    int count = num_neighbours(cells, row, col);
    if (is_alive) {
        if (count == 2 || count == 3) return true;
    } else {
        if (count == 3) return true;
    }

    return false;
}

/**
 * @brief This updates the old cells into the new cells.
 * 
 * @param old_cells 
 * @param new_cells 
 */
void get_new_state(const Cells& old_cells, Cells& new_cells) {
    for (int row = 0; row < new_cells.size(); ++row) {
        for (int col = 0; col < new_cells[0].size(); ++col) {
            new_cells[row][col] = get_new_state(old_cells, row, col);
        }
    }
}

/**
 * @brief Randomly initialises the world.
 * 
 * @param w 
 * @param h 
 * @return Cells 
 */
Cells initial_state(int w, int h) {
    // start randomly
    Cells cells(h, std::vector<bool>(w));
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            cells[i][j] = rand() % 2 == 0;
        }
    }
    return cells;
}

int main() {
    sf::Clock clock = sf::Clock();
    sf::Time previousTime = clock.getElapsedTime();
    sf::Time currentTime;

    // how many cells do we want
    int W = 200;
    int H = 200;

    // create the window
    sf::RenderWindow window(sf::VideoMode::getFullscreenModes()[0], "05 - Game of Life");

    // how big is each cell
    float tile_width = (float)window.getSize().x / W;
    float tile_height = (float)window.getSize().y / H;

    // our current and buffer cells
    Cells current = initial_state(W, H);
    Cells buffer = initial_state(W, H);

    // the rectangles to draw
    std::vector<sf::RectangleShape> rectangles(H * W, sf::RectangleShape({tile_width, tile_height}));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            // set their positions on the grid
            rectangles[i * W + j].setPosition(j * tile_width, i * tile_height);
        }
    }

    while (window.isOpen()) {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event)) {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(sf::Color::Black);

        // update the state
        get_new_state(current, buffer);
        // swap the buffers, so that current contains the updated data in buffer.
        std::swap(current, buffer);
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                // make the rectangles grey (alive) or black (dead)
                rectangles[i * W + j].setFillColor(current[i][j] ? sf::Color(127, 127, 127) : sf::Color::Black);
            }
        }
        // draw them all.
        for (auto& r : rectangles) {
            window.draw(r);
        }

        // end the current frame
        currentTime = clock.getElapsedTime();
        float fps = 1.0f / (currentTime.asSeconds() - previousTime.asSeconds());
        window.setTitle("05 - Game of Life. FPS = " + std::to_string((int)round(fps)));
        previousTime = currentTime;
        window.display();
    }

    return 0;
}
