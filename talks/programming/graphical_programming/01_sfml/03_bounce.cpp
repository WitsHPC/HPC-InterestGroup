#include <SFML/Graphics.hpp>
#include <cmath>

int main()
{
    double position = 0;
    double speed = 2;
    // create the window
    sf::RenderWindow window(sf::VideoMode::getFullscreenModes()[0], "03 - Bouncing");
    // create a circle shape with radius = 100
    sf::CircleShape circle = sf::CircleShape(100);
    // set the origin as the center, instead of the default top left corner.
    circle.setOrigin(circle.getRadius(), circle.getRadius());
    
    // change it's position to the middle of the screen
    position = window.getSize().x / 2;
    circle.setPosition(position, window.getSize().y/2);
    // change its colour to  RGB = 100, 250, 50 = a bright green.
    circle.setFillColor(sf::Color(100, 250, 50));
    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // clear the window with black color
        window.clear(sf::Color::Black);
        position = position + speed;
        // reverse direction if reached either wall.
        if (position + circle.getRadius() >= window.getSize().x || position - circle.getRadius() <= 0){
            speed *= -1;
        }
        circle.setPosition(position, sin(position / 100) * circle.getRadius() + window.getSize().y / 2);
        // draw everything.
        window.draw(circle);

        // end the current frame
        window.display();
    }

    return 0;
}
