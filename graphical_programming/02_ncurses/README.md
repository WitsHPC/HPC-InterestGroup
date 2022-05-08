# Ncurses Introduction
<p align="center">
![](images/matrix.gif)
</p>


**To use this, just run**
```bash
# Optional
sudo apt install libncurses-dev
make
./main
```

Also see [here](https://github.com/Michael-Beukman/curses-game-of-life) for more about the second file `main_game_of_life.cpp`. This can be run using
```
make life
./life
```

# Introduction

-  Today we'll be going over some aspects of graphical programming 
-  As opposed to text-based, cli programs, these programs have a visual element 
-  We will specifically be using ncurses today - visual library for the terminal 



---

# What is ncurses?

-  Ncurses is a UI library for the terminal 
-  Can be used to make a normal app 
-  Or games or animations, which we will focus on today 


---

# Today - Agenda

-  Installation 
-  Basics 
-  Cool Example 


---

# Installation


```bash
sudo apt install libncurses-dev
```




Or, from source like last week.




---

# General Flow

<div>

```cpp
#include <curses.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // need to set this locale to make some unicode characters work
    setlocale(LC_ALL, "");
    cbreak();
    initscr();
    noecho();
    clear();
    curs_set(0);
    while (1){
        erase();
        // draw
        refresh();
    }
    endwin();
    exit(0);
}
```




---

# Usage -> Makefile, source file, simple example

 Compile Using 

```bash
g++ -std=c++11  main.cpp -lncurses -o main
``` 



 Or, use a makefile

```makefile
# makefile
main: main.cpp
	g++ -std=c++11  main.cpp -lncurses -o main
``` 
and just type
```
make
```


---

# Usage -> More Complex Example


Generally, the main useful thing is 

```cpp
mvaddch(y, x, '#');
mvaddstr(y, x, "Hey");
```




Further, you can use colours

```cpp
#define PAIR_GREEN 1

// after the initialisation
start_color();
// name, foreground, background
init_pair(PAIR_GREEN, COLOR_GREEN, COLOR_BLACK); 

// later, before drawing
attron(COLOR_PAIR(PAIR_GREEN));
```



---

# Demo / Example

Live Eample / Demo, see `main.cpp`
```cpp
#include <curses.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>

#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#define ITERMAX 10000
#define PI 3.14159265359f
#define PAIR_GREEN 1
#define PAIR_WHITE 2

/**
 * @brief Returns the current number of microseconds since epoch
 *
 * @return int
 */
int current_microseconds()
{
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() * 1000;
}

/**
 * @brief Returns a random string
 * @return string
 */
std::string get_str()
{
    std::string str = "";
    if ((rand() % 10) < 7)
        str += " ";
    else
    {
        if (rand() % 2 == 0)
            str += (rand() % 26) + 65;
        else if (rand() % 2 == 0)
            str += (rand() % 26) + 65 + 32;
        else
            str += (rand() % 12) + 33;
    }
    return str;
}

/**
 * @brief Returns a random string of length LINES
 * @return string
 */
std::string get_full_string()
{
    std::string str = "";
    for (int r = 0; r < LINES; ++r)
    {
        str += get_str();
    }
    return str;
}

/**
 * @brief Takes in a vector of strings, and a vector of offsets per column.
 * This draws the text in columns, and adjusts the offsets to get an animation effect.
 */
void draw_matrix_lines(std::vector<std::string> &all_texts, std::vector<float> &all_offsets)
{
    // for all columns
    for (int c = 0; c < all_texts.size(); ++c)
    {
        bool hasdone = false;
        // for all rows
        for (int r = 0; r < all_texts[c].length(); ++r)
        {
            if (r + (int)all_offsets[c] < LINES && r + (int)all_offsets[c] >= 0)
            {
                // set colour
                if (!hasdone)
                    attron(COLOR_PAIR(PAIR_WHITE));
                // draw the text
                mvaddch(r + (int)all_offsets[c], c, all_texts[c][(r) % all_texts[c].length()]);

                // set the colour back to green
                if (!hasdone)
                {
                    hasdone = true;
                    attron(COLOR_PAIR(PAIR_GREEN));
                }
            }
        }
        // update the offsets
        if (c == 0)
            all_offsets[c] += 1;
        else
            all_offsets[c] += 0.2 * (abs(sin(c)) + (float)(rand() % 10) / 40);

        if (all_offsets[c] >= LINES)
        {
            all_offsets[c] = -LINES;
            // add new random text.
            all_texts[c] = get_full_string();
        }
    }
}

int main(int argc, char **argv)
{
    std::string character = "#";
    int speed = 30;

    // need to set this locale to make some unicode characters work
    setlocale(LC_ALL, "");

    // Init
    cbreak();
    initscr();
    noecho();
    clear();
    curs_set(0);
    if (has_colors() == FALSE)
    {
        endwin();
        printf("Your terminal does not support color\n");
        exit(1);
    }
    // Make the colours
    start_color();
    init_pair(PAIR_GREEN, COLOR_GREEN, COLOR_BLACK);
    init_pair(PAIR_WHITE, COLOR_WHITE, COLOR_BLACK);

    std::vector<std::string> all_texts; // Each column is a string
    std::vector<float> all_offsets;     // Each Str is a string

    for (int c = 0; c < COLS; ++c)
    {
        all_offsets.push_back(0);
        all_texts.push_back(get_full_string());
    }

    int i = 0;
    float angle = 0;
    // Used to get a consistent frame rate
    int time_now = current_microseconds();
    attron(COLOR_PAIR(PAIR_GREEN));
    while (1)
    {
        ++i;
        // clear();
        erase();

        // Draw
        draw_matrix_lines(all_texts, all_offsets);

        refresh();
        // update the new array

        int new_time = current_microseconds();
        // want to sleep 1000 / speed milliseconds
        int time_to_sleep = std::max(1e6 / speed - (new_time - time_now), 0.0);
        usleep(time_to_sleep);
        time_now = current_microseconds();
    }
    endwin();
    exit(0);
}
```

---

# Next Steps


-  Read more about ncurses 
-  Build your own fun projects! 
   - See [here](https://github.com/Michael-Beukman/curses-game-of-life) for another (slightly more complex) example.

---


# End, Thanks!
Sources

-  https://en.wikipedia.org/wiki/Ncurses 
-  https://tldp.org/HOWTO/NCURSES-Programming-HOWTO/intro.html 

Inspiration:
- https://www.youtube.com/watch?v=kqUR3KtWbTk
- https://www.youtube.com/watch?v=z8Li_gQyeiM
- https://en.wikipedia.org/wiki/Matrix_digital_rain