## Compiling ncurses from Source
By: Sayfullah Jumoorty, with more instructions [here](https://github.com/mirror/ncurses/blob/master/INSTALL).
1. **Download the source code**:
     ```shell
     wget https://invisible-island.net/datafiles/release/ncurses.tar.gz
     ```

2. **Extract the source code**:
   - Open your terminal or command prompt.
   - Navigate to the directory where you downloaded the tarball (e.g., `cd ~/Downloads`).
   - Extract the tarball using the following command:
     ```shell
     tar xf ncurses.tar.gz
     ```

3. **Configure the build**:
   - Change into the extracted directory (e.g., `cd ncurses-x.x`).
   - Run the configure script with your desired options. For example:
     ```shell
     ./configure --prefix=/your/directory/here
     ```
     This command specifies the installation prefix, which can be customized according to your needs.

4. **Build the ncurses library**:
   - Execute the `make` command to compile the source code:
     ```shell
     make
     ```

5. **Install ncurses**:
   - Run the following command to install ncurses:
     ```shell
     make install
     ```
     This will install the ncurses library and related utilities into the specified prefix (e.g., `/usr/local/ncurses`).
     
6. **Install ncurses**:
   - Add the built library paths to your bashrc file, replace x.x with your version shown on your folder
     ```shell
     export PATH=/your/directory/here/ncurses-x.x/bin:$PATH
     export LD_LIBRARY_PATH=/your/directory/here/ncurses-x.x/lib:$LD_LIBRARY_PATH
     ```
     This will add the ncurses library and related utilities into the specified path.

7. **Verify the installation**:
   - To ensure that ncurses is correctly installed, compile a simple program that uses ncurses. Create a file named `test.c` with the following contents:
     ```c
     #include <ncurses.h>

     int main() {
         initscr();              // Initialize ncurses
         printw("Hello, ncurses!"); // Print a message
         refresh();              // Refresh the screen
         getch();                // Wait for user input
         endwin();               // Clean up and exit ncurses
         return 0;
     }
     ```
   - Compile the program using the installed ncurses library:
     ```shell
     gcc -o test test.c -lncurses
     ```
   - Execute the compiled program:
     ```shell
     ./test
     ```
   - If you see the message "Hello, ncurses!" displayed in the terminal and it waits for user input before exiting, then ncurses is working correctly.

That's it! You have successfully compiled and installed ncurses from source.