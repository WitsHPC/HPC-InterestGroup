# More Advanced Linux Shell Commands
## General Commands

In Linux/Unix, basically everything is a file, so the following commands are super useful.

- `ls` **L**i**s**t all files in a directory. Usage is either `ls` to list the files in the current dir, or `ls dir` to list the files in a directory (e.g. `ls /home/user/`)
    - `ls -l` (list in a table format with more information)
    - `ls -lt` (Sorts based on time, most recently changed is at the top)
    - `ls -lh` (Makes the file sizes **h**uman readable, i.e. 10KB instead of 10035)
- `cd` **C**hange **D**irectory: This changes your current directory.
    - `cd`: Change directory to the home directory
    - `cd -` Change directory to the previous one
    - `cd dir` navigate to the `dir` directory.
    - `cd ..` → CD one directory up the tree
- `pwd` **P**rints the current (**w**orking) **d**irectory
- `cat` Displays a file. Usage (among others): `cat filename`
- `cp` **C**opies a file or directory. Usage:
    - `cp source destination`
    - `cp -r source_directory destination_directory`
- `mv` **M**o**v**es, or renames a directory or file. Usage: `mv source dest`
- `rm` **R**e**m**oves a file or directory. Usage
    - `rm file`
    - `rm -r directory`
- `head` Displays the top `n` lines of a file (defaults to 10). Usage `head -n 20 file` or `head file`
- `tail` Similar to `head`, just displays the last `n` lines.
- `less` Allows you to read a long file, with scrolling (mouse movement) and searching (type `/` and the search word)
- `touch` Usage is `touch file` and it does the following:
    - If the file exists, update it's modified time
    - If it doesn't exist, create a new empty file with that name.
- `grep`: Search for text in a file
    - Usage (one of many) is `grep myword myfile` → Will print all lines in `myfile` that contain 'myword'
- `echo`: Will print out something. For example, `echo 1` will print out 1.
- `mkdir`: Will create the directory specified, e.g. `mkdir my_directory`
    - Shortcut if you want to create nested directories: `mkdir my_directory/subdir` will fail, but `mkdir -p my_directory/subdir` will work

- `find .`
  - Finds files in a directory. Discussed more later


Using these commands, you can navigate the directory structure, read files, copy and move them, and see where you are.

You can run two commands in one line using either:

- `command1; command2` → both commands will run, even if one of them fails
- `command1 && command2` → command2 will only run if command1 did not give an error

## Redirection

### Pipes

One super cool thing about linux is you can combine the above commands using the `|` (pipe) symbol.

This works as follows:

`command1 | command2`, which will take the output of command1 and give it to command2 as input.

e.g. `ls -lt | head` → Will do an `ls -lt`, and return the first 10 lines

This is basically equivalent to you doing the following:

copying the output of `ls -lt` and pasting it into a file named `temp`

calling `head temp`, but without you having to do any manual work.

Some very common use cases are:

- Basically anything with `grep` at the end (e.g. `ls -l | grep .myextension`)
    - Or grepping multiple things per line, e.g. `grep A file | grep B` will return lines that contain both an A and a B
- using `less`, `head` or `tail` at the end to either read a long piece of text, or just view the end or beginning of it.

### Redirecting to files

There is another part of redirection, redirecting to files.

for example, `ls -l > myfile.txt` will save the output of `ls -l` in a file called `myfile.txt`, and overwrites it if it already exists

Some other ones: 

- `>>`: Append to the file (i.e. if it already exists, it only adds to the end and doesn't overwrite what's already there)
- `2>` Direct the error output to a file
- `&>` Direct the error and normal output to a file

And, if you want to see the output **and** have it go to a file, use the `tee` command:

Usage is something like `ls -l | tee myfile.txt` → the output will be shown in the terminal and also written to a file (use `tee -a` to append instead of overwriting)

You can also use backticks (`) to use the output of a command as the arguments of another one.

For example, `find . | ls -lth` doesn't list all the files returned by find, but only the files in the current directory.

You can get around this by running 
```
ls -lth `find .`
```
## Directories

Directories are containers for other directories and files

There are some 'special' directories that are quite useful

`.` is always the current directory

`..` is always one directory up the tree

`~` is the home directory of the current user, e.g. `/home/<yourname>`

The directory structure (in linux) is usually something like this:

`/` → The root directory, everything else is contained in here

`/root` → The home directory of the root user (i.e. admin)

`mnt` → Usually where devices (e.g. USBs) are mounted, and where you can access them

`/home` → The home directory, contains one folder for each user, e.g. `/home/john` , `/home/bob`

## Searching

There are a few common commands to find stuff, either inside files, or find files themselves.

The main ones here are `find` and `grep`.

`find` returns a list of all the filenames in a specific directory, that satisfy some criteria

For example, `find dir` will return a list of all files and directories in `dir`

There are many options (see `man find` for a full list), but some useful ones are:

- `-type f` or `-type d` to return only files (f) or directories (d)
- `-name regex` returns all files or directories that have a specific pattern in their name
    - This pattern can be a simple name, but it can also include wildcards, like `*` (any number of any character) and `?` (one single character)
    - e.g. `find . -name myfile*.txt` will match myfile.txt, myfile1.txt, myfile1234.txt, myfiles.txt, etc.
    - `find . -name myfile?.txt` will match myfile1.txt, myfiles.txt, but not the others.

`grep` is arguably used more often, and it can extract lines in a file that meet a certain criteria, with  similar (but slightly different) pattern wildcards as find.

Usage: `grep <pattern> file` or `grep '<pattern>' file`

Other useful options:

- `-i` → be case insensitive
- `-v` → return only lines that don't match the pattern
- `-r` → search all files recursively in a directory

Some other useful patterns:

- `^` matches the beginning of a line, so `grep '^Hello' file` will return all lines that begin with 'Hello'
- `$` matches the end of a line
