# nSnake

Compiling a fun snake game in the terminal.

## Download and Extract

We will first download the source code using the `curl` command. We will download it as tarball (like a .zip file).

```bash
curl -L https://github.com/alexdantas/nSnake/archive/refs/heads/master.tar.gz -o nSnake-master.tar.gz
```

We will then extract the tarball using the `tar` command.

> [!NOTE]
> the `tar` command takes some "flags" (options for a command). The `-xf` is a combination of two flags `-x` which means to "extract" and `-f` which means "file". More on `tar` [flags](https://www.gnu.org/software/tar/manual/html_section/All-Options.html).

```bash
tar -xf nSnake-master.tar.gz
```

## Compile

The `make` executes the contents of a `Makefile`. The `Makefile` is the "how you want to compile file". For more on [Makefiles](https://opensource.com/article/18/8/what-how-makefile).

```bash
make
```

## Run in the file

> [!NOTE]
> The `./` will run a executable "application".

```bash
./bin/nsnake
```

## Add to environment to run anywhere (wow)

So now we can run nSnake; but we cant run it in any folder, we will achieve this by telling our environment where an executable is "the location". We will use the `~/.bashrc` file to achieve this.

> [!NOTE]
> The `~/.bashrc` file tells the terminal what environment to always set to the terminal. Lik where applications are, commands to run when a new terminal instance is made, etc.

```bash
vim ~/.bashrc
```

Add to the end of the file the following:

```bash
export PATH=$PATH:<PATH_TO_FOLDER>/bin/
```

> [!IMPORTANT]  
> To exit and save vim press `ESC` and then type `:wq` (write & quite) and then click `ENTER` :)

Then to apply the environment script we will use the `source` command:

```bash
source ~/.bashrc
```

Go to your home directory "any directory would do" and run run nSnake

```bash
cd ~
nsnake
```
