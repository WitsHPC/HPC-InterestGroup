#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
    const double MAX_FORCE = 400.0;
    sf::Vector2<double> position = {0, 0};
    sf::Vector2<double> speed = {0, 0};
    // create the window
    sf::RenderWindow window(sf::VideoMode::getFullscreenModes()[0], "04 - Interaction");
    // create a circle shape with radius = 100
    sf::CircleShape circle = sf::CircleShape(100);
    // set the origin as the center, instead of the default top left corner.
    circle.setOrigin(circle.getRadius(), circle.getRadius());

    // change it's position to the middle of the screen
    position.x = window.getSize().x / 2;
    position.y = sin(position.x / 100) * circle.getRadius() + window.getSize().y / 2;

    circle.setPosition(position.x, position.y);
    // change its colour to  RGB = 100, 250, 50 = a bright green.
    circle.setFillColor(sf::Color(100, 250, 50));
    bool is_clicked = false;
    // run the program as long as the window is open

    sf::Vector2i mousepos_prev;
    while (window.isOpen()) {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event)) {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::MouseButtonPressed) {
                // right mousebutton, stop the ball
                if (event.mouseButton.button == sf::Mouse::Right) {
                    speed *= 0.0;
                // this is if we pressed the left btn, so we set is_clicked = true, to indicate that we are dragging atm.
                } else if (!is_clicked) {
                    mousepos_prev = sf::Mouse::getPosition(window);
                    is_clicked = true;
                }
            }

            if (event.type == sf::Event::MouseButtonReleased) {
                // if we released the left button
                if (event.mouseButton.button == sf::Mouse::Left) {
                    // not dragging anymore
                    is_clicked = false;
                    
                    // vector from the mouseposition to the ball
                    auto diff_vec = sf::Vector2<double>(position - sf::Vector2<double>(sf::Mouse::getPosition(window)));
                    // clamp the norm to not be too long
                    double norm = sqrt(pow(diff_vec.x, 2) + pow(diff_vec.y, 2));
                    if (norm >= MAX_FORCE) {
                        diff_vec = diff_vec / norm * MAX_FORCE;
                    }
                    // increase speed with the 'force' (= acceleration if the mass is 1)
                    speed += diff_vec / 200.0;
                }
            }
        }

        // clear the window with black color
        window.clear(sf::Color::Black);
        position = position + speed;
        
        // Just check if either the x or y positions are out of bound, and bounce off then.
        if (position.x + circle.getRadius() >= window.getSize().x || position.x - circle.getRadius() <= 0) {
            position.x = std::clamp(position.x, (double)circle.getRadius() + 1, (double)window.getSize().x - circle.getRadius() - 1);
            speed.x *= -1;
        }

        if (position.y + circle.getRadius() >= window.getSize().y || position.y - circle.getRadius() <= 0) {
            position.y = std::clamp(position.y, (double)circle.getRadius() + 1, (double)window.getSize().y - circle.getRadius() - 1);
            speed.y *= -1;
        }

        circle.setPosition(position.x, position.y);
        // if we have clicked, draw a line from the mouse position to the ball to indicate a direction.
        if (is_clicked) {
            sf::Vertex line[] =
                {
                    sf::Vertex(sf::Vector2f(position)),
                    sf::Vertex(sf::Vector2f(sf::Mouse::getPosition(window)))};

            window.draw(line, 2, sf::Lines);
        }
        // draw everything.
        window.draw(circle);

        // end the current frame
        window.display();
    }

    return 0;
}
