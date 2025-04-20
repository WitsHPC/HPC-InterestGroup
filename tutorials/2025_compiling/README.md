# nSnake

A fun snake game in the terminal.

## Download and Extract

```bash
curl -L https://github.com/alexdantas/nSnake/archive/refs/heads/master.tar.gz -o nSnake-master.tar.gz
```

```bash
tar -xf nSnake-master.tar.gz
```

## Compile

```bash
make
```

## Run in the file

```bash
./bin/nsnake
```

## Add to environment to run anywhere (wow)

```bash
vim ~/.bashrc
```

Add to the end of the file:

```bash
export PATH=$PATH:<PATH_TO_FOLDER>/bin/
```
To exit vim `ESC` and then type `:wq` and then click `ENTER` :)

Then to apply the environment script

```bash
source ~/.bashrc
```

Go to your home directory and run run nSnake

```bash
cd ~
nsnake
```
