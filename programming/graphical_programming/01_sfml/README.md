# Graphical Programming
- [Graphical Programming](#graphical-programming)
- [Introduction](#introduction)
- [Great sources / frameworks](#great-sources--frameworks)
    - [C / C++](#c--c)
    - [Java](#java)
    - [JavaScript](#javascript)
    - [Python](#python)
- [Why?](#why)
- [Examples](#examples)
- [How to](#how-to)
  - [Basics](#basics)
  - [SFML](#sfml)
    - [Installing](#installing)
    - [Hello World](#hello-world)
    - [Motion](#motion)
    - [Interaction](#interaction)
    - [Pixels](#pixels)
    - [Final - Game of Life](#final---game-of-life)
- [Next Steps](#next-steps)
- [Sources](#sources)
# Introduction

In this talk we'll be covering graphical programming, i.e. writing code that displays visual elements on screen. This can often be interactive, colourful and animated.

# Great sources / frameworks

First of all, you can do this type of stuff in most languages, but some make it easier than others.

I personally have experience with doing this type of thing in C++, JavaScript, Java and Python, but in principle other languages could work too. Some are more difficult to get up and running though.

Usually you need a framework to actually do things, and here are a few ones I can recommend:

### C / C++

- [SDL](https://www.libsdl.org/) → Old school C library that can work with C++
- [SFML](https://www.sfml-dev.org/) → Newer, C++ framework for 2D graphics that makes it super easy to get up and running.

### Java

- [https://processing.org/](https://processing.org/) → Creative coding framework that is made for visual programming, and non programmers, so it's very easy to learn, but you can still do very cool things with it, including 3D graphics.

### JavaScript

- [https://p5js.org/](https://p5js.org/) → Very easy to get started with, you can use their web editor, and all things you make can easily be showcased through the web. Basically a clone of Processing for the web. **If you are unsure about what to use, go for this**
- [https://threejs.org/](https://threejs.org/) → Great 3D JS library.

### Python

- [https://www.pygame.org/](https://www.pygame.org/) → PyGame, specifically made for creating games in Python
- [http://pyglet.org/](http://pyglet.org/) → Python graphics library, pretty useful.

If you want to get more into this area, then I can wholeheartedly recommend the following youtube channels:

- [Javidx9](https://www.youtube.com/channel/UC-yuWVUplUJZvieEligKBkA) → Awesome videos in C++, goes over basically everything, and how to build a game engine in a windows cmd line, how to do 3D graphics from scratch, and other great videos.
- [The Coding Train](https://www.youtube.com/c/TheCodingTrain) → Has great tutorials on Processing, P5.js, learning how to code, as well as many other cool visualisations, as well as interesting concepts like genetic algorithms, physics, machine learning etc.

# Why?

Why would you want to do this? First of all, it is lots of fun, but it can also be very useful in that you can also visualise algorithms to understand / explain them better, investigate cool mathematical patterns, and more.

You could also make games or other interactive applications.

# Examples
Here are a few examples from other people's work.

P5 Showcase → [https://showcase.p5js.org/#/2020-All](https://showcase.p5js.org/#/2020-All)


![Untitled](images/Untitled.png)

[https://www.youtube.com/watch?v=xW8skO7MFYw](https://www.youtube.com/watch?v=xW8skO7MFYw) → Original DOOM like rendering of 2d scenes into pseudo 3D.

![Untitled](images/Untitled%201.png)

[https://www.youtube.com/watch?v=Y37-gB83HKE](https://www.youtube.com/watch?v=Y37-gB83HKE) → Make and solve mazes

![Untitled](images/Untitled%202.png)

[https://youtube.com/watch?v=FGAwi7wpU8c](https://youtube.com/watch?v=FGAwi7wpU8c) → Solar System Simulation

![Untitled](images/Untitled%203.png)

[https://youtube.com/watch?v=fAsaSkmbF5s](https://youtube.com/watch?v=fAsaSkmbF5s) → Julia Set

# How to

In this talk we'll be mainly going over SFML, as that is in C++, and we can apply some of our performance principles and parallel programming techniques easily, but you can basically do the same in any of the other frameworks.

## Basics

In graphics, we have this concept of the screen space, specifically a coordinate system that is bounded by (0, 0) and (w, h). The Y axis is points downward, and the origin is at the top left.

![Untitled](images/Untitled%204.png)

In addition to this, we have the concept of a pixel, a single block that we can assign a colour. The above coordinate system is in terms of pixels, so we have w * h pixels in total.

A colour can be represented using a **R**ed value, **G**reen value and a **B**lue value, usually integers between 0 and 255. You could also sometimes have a transparency value, called alpha.

Usually, most drawing applications will have this concept of a 'game loop', which provides you opportunity to do any processing, and draw a new frame to the screen.

## SFML

### Installing

You can follow the steps here: 

[https://www.sfml-dev.org/download/sfml/2.5.1/](https://www.sfml-dev.org/download/sfml/2.5.1/)

to download and install SFML. If you have any issues, then feel free to ask for help.

### Hello World

Code: `01_hello.cpp`

There are a few key parts when using SFML, specifically:

- Creating a window
- The event loop
- Drawing stuff

We can create a window using. We could create a full screen one like this

```cpp
sf::RenderWindow window(sf::VideoMode::getFullscreenModes()[0], "01 - Hello World");
```

Or a 400 x 400 one like this:

```cpp
sf::RenderWindow window(sf::VideoMode(400, 400), "01 - Hello World");
```

Then, we need to loop, and perform some operations while the window isn't closed yet.

We can do this using the following:

```cpp
// run the program as long as the window is open
while (window.isOpen()) {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event)) {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }
}
```

And then on to the most important part - drawing things.

There are a few main ways to draw in SFML, but the simplest one is the `Shape`

There are a few built in shape types, like

- `RectangleShape`
- `CircleShape`
- Regular polygons.
    - These can be created using the second argument in the constructor of `CircleShape`, which gives the number of edges. A circle is basically a polygon with infinite edges.
    - `sf::CircleShape triangle(80.f, 3);`

We can also transform these, to change their position, scale and origin, or change the fill and outline colour.

We could for example use the following to move the circle to the middle of the screen

```cpp
circle.setPosition(window.getSize().x / 2, window.getSize().y / 2);
```

For the above to work properly, we first set the origin of the circle to its centerpoint. 

The origin of a shape is by default the top left, and all transformations are relative to this point

```cpp
circle.setOrigin(circle.getRadius(), circle.getRadius());
```

We could change the colour using:

```cpp
circle.setFillColor(sf::Color(100, 250, 50));
```

And then in our loop, we can draw it using

```cpp
// clear the window with black color
window.clear(sf::Color::Black);

// draw everything.
window.draw(circle);
```

### Motion

Code: `02_motion.cpp` and `03_bounce.cpp`

We can do motion by simply having a position variable that we update at every frame (i.e. iteration of the while loop). We then update the shape's position with this value and draw it.

Before the while loop.

```cpp
double position = window.getSize().x / 2;
circle.setPosition(position, window.getSize().y/2);
```

And in the while loop:

```cpp
// clear the window with black color
window.clear(sf::Color::Black);
// change the position, taking care to wrap around
position = fmod(position + speed, window.getSize().x);
// set the position
circle.setPosition(position, sin(position / 100) * circle.getRadius() + window.getSize().y / 2);
// draw
window.draw(circle);
```

### Interaction

Code: `04_interaction.cpp`. Drag the ball and release the mouse to shoot it in a direction.

So the above is nice and all but sometimes you want some interaction, where the user can affect what is happening here.

The main way we do this in SFML is to use the event polling system.

Inside our `while (window.isOpen()) {` loop, we can put the following in:

```cpp
while (window.pollEvent(event)) {
		// "close requested" event: we close the window
      if (event.type == sf::Event::Closed){
            window.close();
			}
        if (event.type == sf::Event::MouseButtonPressed) {
						// a mouse button was pressed.
        }

        if (event.type == sf::Event::MouseButtonReleased) {
            if (event.mouseButton.button == sf::Mouse::Left) {
								// the left mousebutton was released
            }
        }
    }
```

In the code we keep track of a difference vector between the ball and the current mouse position. When the mouse is released, we apply a force to the ball to make it move.

### Pixels

code: `05_game_of_life_pixels.cpp`

Often we want to manipulate the individual pixels on the screen, and SFML provides a way to do this.

*Warning: If you are still drawing shapes, then it will often be more performant to just use the SFML shape routines. Only use this if you really want to draw everything yourself.*

It's often useful to use rectangles instead, so you can change the size of a 'virtual pixel', so one rectangle could be 8x8, but you consider it as the atomic, single pixel.

We can have single pixel access though, by using a texture, which is basically just some pixels that represent an image that covers a specific sprite, and the sprite to hold the texture and actually get drawn to the screen.

```cpp
sf::Sprite sprite;
sf::Texture tex;
// create the texture with a specific width and height
tex.create(width, height);
sprite.setTexture(tex);
```

We also need a vector to actually store the pixels in. We use the type `uint8`, to store colour values between 0 and 255. We need 4 of these values (RGBA) per pixel.

We also use a 1D array instead of a 2D one, as that is what SFML expects, but we can easily index the k-th colour channel (k is between 0 and 3) of  pixel `i, j` using `pixels[4*(i * width + j) + k]`

```cpp
// make a vector that represents our pixels. width * height pixels, 
// each pixel has a value for RGBA
std::vector<sf::Uint8> pixels(width * height * 4, 255);
```

At each loop iteration, we update the texture with the pixels

```cpp
// update the pixels array.
        int col;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // each pixel has 4 numbers, so multiply the index by 4.
                int index = (y * width + x) * 4;
                col = 127 * current[y][x];
                // update the RGB values. The alpha values have a default of 255, and we don't change that.
                pixels[index + 0] = col; // R
                pixels[index + 1] = col; // G
                pixels[index + 2] = col; // B
            }
        }
```

And draw the sprite.

```cpp
// Update the texture from the pixels data.
tex.update(pixels.data());
// draw the sprite.
window.draw(sprite);
```

![mygif.gif](images/mygif.gif)

### Final - Game of Life

code: `06_game_of_life.cpp`

This is more performant because we use larger rectangles instead of individual pixels, but it still does the same thing.

![gif1.gif](images/gif1.gif)

# Next Steps

Now, you should be able to draw many different things, but to get fluent and more comfortable, you really need to practice these things. 

I'd suggest trying to visualise something, like the next algorithm / mathematical thing you learn in university, or making a simple game, or literally anything with fractals.


*Suggestions / Fixes are welcome*
# Sources

- [https://www.sfml-dev.org/](https://www.sfml-dev.org/)
- [https://www.sfml-dev.org/tutorials/2.5/](https://www.sfml-dev.org/tutorials/2.5/)
- [https://en.wikipedia.org/wiki/Conway's_Game_of_Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- [https://math.hws.edu/graphicsbook/c2/s1.html](https://math.hws.edu/graphicsbook/c2/s1.html)
- [https://en.sfml-dev.org/forums/index.php?topic=16547.0](https://en.sfml-dev.org/forums/index.php?topic=16547.0)