# nSnake

> Original by Clare Codeiro and Talhah Patelia  
> Changes by Lily de Melo

Compiling a fun snake game in the terminal using a shell script.

If you are using your own PC you need to install:
```
sudo apt install libncurses-dev
```

Instead of running every command one at a time, we will add each step to a
`.sh` file. A shell script lets us save terminal commands and run them again
later.

## Create the Script

Create a new script file called `install_nsnake.sh`.

```bash
vim install_nsnake.sh
```

Add the first two lines to the `.sh` file:

```bash
#!/bin/bash
set -e
```

The first line tells the computer to run this file using `bash`. The second
line stops the script if one of the commands fails.

## Download the Source Code

Add this command to the `.sh` file to download the nSnake source code:

```bash
curl -L https://github.com/alexdantas/nSnake/archive/refs/heads/master.tar.gz -o nSnake-master.tar.gz
```

Your script should now look like this:

```bash
#!/bin/bash
set -e

curl -L https://github.com/alexdantas/nSnake/archive/refs/heads/master.tar.gz -o nSnake-master.tar.gz
```

## Extract the Download

Add this command to the `.sh` file to extract the downloaded tarball:

```bash
tar -xf nSnake-master.tar.gz
```

## Move Into the Folder

Add this command to the `.sh` file so the script enters the extracted source
code folder before compiling:

```bash
cd nSnake-master
```

## Compile nSnake

Add this command to the `.sh` file to compile nSnake:

```bash
make
```

## Run nSnake From the Script

Add this command to the `.sh` file to run nSnake after it compiles:

```bash
./bin/nsnake
```

## Full Script

Your `install_nsnake.sh` file should now contain:

```bash
#!/bin/bash
set -e

curl -L https://github.com/alexdantas/nSnake/archive/refs/heads/master.tar.gz -o nSnake-master.tar.gz
tar -xf nSnake-master.tar.gz
cd nSnake-master
make
./bin/nsnake
```

Save and close the file.

## Make the Script Executable

Run this command in the terminal to allow the `.sh` file to run like a program:

```bash
chmod +x install_nsnake.sh
```

## Run the Script

Run the `.sh` file:

```bash
./install_nsnake.sh
```

The script will download, extract, compile, and run nSnake.

## Optional: Run nSnake Later

After the script has finished compiling nSnake once, you can run the game again
from inside the `nSnake-master` folder:

```bash
cd nSnake-master
./bin/nsnake
```
