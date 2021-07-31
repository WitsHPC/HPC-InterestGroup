# Shell Intro
# Introduction

Unix is an operating system that started in the 1970s. Many current operating systems are based off of Unix, notably Linux and Apple's macOS. 

Linux itself has many distributions, like CentOS and Red Hat (which is often used in HPC), Ubuntu / Mint, which are usually more consumer desktop / laptop focused.

You can usually type in commands into a program called a shell, which you can access using the terminal. These commands let you do a variety of things, from reading and editing files to changing system settings, to getting diagnostic information.

There are many different shells, but the most common one is [bash](https://www.gnu.org/software/bash/), but other improved versions include [zsh](https://www.zsh.org/) and [fish](https://fishshell.com/).

Basically everything we will be covering here is applicable to all of the above, but we'll be focusing on the bash shell, and there are some commands that do not carry over exactly to other shells.

Note: This is not an exhaustive guide and it merely attempts to give a brief outline of some common commands and how to combine them to perform useful functions.

For a more in depth discussion, definitely have a look at the sources listed at the end.

The structure of this talk will be as follows:

1. Get started and some basic commands
2. Basic files and how they work
3. How commands take input and how to combine multiple commands.
4. Searching
5. Transformation
6. File structure and editing. 

# Get Started

To actually use a shell, you can do the following:

- **Linux**: Just right click on the desktop and select open in terminal
- **Mac**: Just search for the Terminal application and open it. Most commands and techniques are discussed here is applicable to Mac, but the directory
- **Windows**: Install the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10), which allows you to use a linux shell inside windows.

You should now have a screen with a blinking cursor where you can type text in.

If you get stuck, aren't sure what a command does, or are curious about the multiple different options that a command can take in, use the `man` command.

You can use it like: `man ls`, which will give you a manual entry for the `ls` command, and it's usually quite detailed and contains examples.

The best way to become familiar with these commands is to actually use them, so feel free to follow along with a terminal open.

# Files

In Unix, basically everything is a file, so the following commands are super useful.

- `ls` **L**i**s**t all files in a directory. Usage is either `ls` to list the files in the current dir, or `ls dir` to list the files in a directory (e.g. `ls /home/user/`)
    - `ls -l` (list in a table format with more information)
    - `ls -lt` (Sorts based on time, most recently changed is at the top)
    - `ls -lh` (Makes the file sizes **h**uman readable, i.e. 10KB instead of 10035)
- `cd` **C**hange **D**irectory: This changes your current directory.
    - `cd`: Change directory to the home directory
    - `cd -` Change directory to the previous one
    - `cd dir` navigate to the `dir` directory.
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

Using these commands, you can navigate the directory structure, read files, copy and move them, and see where you are.

You can run two commands in one line using either:

- `command1; command2` → both commands will run, even if one of them fails
- `command1 && command2` → command2 will only run if command1 did not give an error

# Files and Input

In linux, most commands have two different ways of taking in input, either using a file or using standard input. Standard input is basically when a program asks for input from the user. For example, you could write a C++ program that does the following:

```cpp
string name;
cout << "Please enter your name\n";
cin >> name;
```

Or, in Python

```python
name = input("Please enter your name")
```

the user types in their names into standard input

For example, the cat command can be invoked as follows:

`cat file`, which will simply print out the file

OR 

`cat`

and then you can type in something and press control + D (i.e. in the same way control + S saves a document) and `cat` will then print out what you typed.

This is useful for the next concept, redirection:

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

You can get around this by running `ls -lth `find .``

# Editing

Since files are such a critical part of linux, it's useful to know how to edit these files, and for this you use a text editor. A few common ones are:

- `nano`  → Simple terminal editor that you can quickly edit a file with. Relatively straightforward to use.
- `vim`    → More advanced editor, that has lots of useful features for quick navigation and editing, but it has a steep learning curve.
- `Visual Studio Code` → Usually you'll only use this if you want to edit lots of files. For small files, the other options are faster and definitely enough for just assorted and ad-hoc file editing.

# Directories and Permissions

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

Each directory (and file therein) has permissions, and these are 3 different permissions, that apply to 3 different user groups:

The permissions are:

- `r` → Can read this file
- `w` → Can write this file
- `x` → Can execute this file. Often this is required for directories to actually read their contents.

And there are 3 different user groups that these can apply to:

`user` (u)    → The owner of the files

`group` (g)  → Part of the group of the current user

`other` (o)   → All other users

You can use `ls -l` to see the permissions. Each entry has 10 characters, representing:

`d / -` → Directory (`d`) or file (`-`)

`rwx` → The permissions of the owner (a `-` indicates that that permission is not available)

`rwx` → The permissions of the group

`rwx` → The permissions of other users

```
-rw-rw-r--  2 mike mike        0 Oct 26  2020 myFile
```

You can change these permissions using the `chmod` command

Usage is 

- `chmod o+rx file` → Gives read and execute permissions to 'others'
- `chmod u-w file` → Removes write permissions from the owner.
- `chmod 755 file` → Gives permissions (rwx to user, rx to group and rx to other)
    - In binary, each permission is a 3 digit number, so rwx = 111 = 7 in decimal
    - rx = 101 = 5

# Searching

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

# Transformation

### Sed

Often we want to transform some text (or output from another command) into a different structure, be it by only selecting some fields, replacing some text with something else, or performing computations.

The main ways to do this is using `sed` and `awk`

`sed` is mostly used for find and replace through the use of regular expressions (see Searching)

Usage is for example: `sed 's/name/Name/g' file` will replace all instances of name with Name

You can use more advanced regular expressions too

There are lots of different options, so feel free to look them up. A quite useful one is `sed -i`, which edits the file in place.

### Awk

The `awk` command is quite useful for many different text processing tasks.

The main idea with awk is that it handles each line separately, and chops each line up into fields. You can then perform any operations using these fields.

For example, if we have a file

`awk.txt`

```bash
1 one eins uno
2 two zwei dos
3 three drei tres
```

We can invoke awk as follows:

- `awk '{print $3}' awk.txt` → Which will result in

    ```bash
    eins
    zwei
    drei
    ```

The structure of the code inside the curly braces is what awk does to each line.

- First it splits up the text in each line using a separator (a space by default)
- Then it sets those to the variables `$1` (for the first field → 1, 2, 3) `$2` (for the second field, one two three), etc.
- So, in the above example, we simply print out the third column for each line.
- `$0` is a special variable that contains all of the fields.

There are many options that you can give to awk, and inside the `{}` you can basically write any `awk` code.

Some notable options are:

- `-F`: The separator, which defaults to space. An example of usage is `awk -F',' ...` (can be more than one character)
- `-f`: Program file: If your program is more complex, you can write in a file, and just tell awk to use that file.

But there are many more, so have a look at the man page, or the `unix_intro` pdf

You can also use the `BEGIN` and `END` keyword to indicate what should happen at the beginning and end.

For example, `awk 'BEGIN {COUNT=0} {COUNT += 1} END {print COUNT}' file` will give the number of lines, which could also be obtained using:

- `wc -l file` → (word count command, `-l` (for lines) flag)
- `awk 'END {print NR}' file` → Special `NR` variable which counts the number of rows

# Useful Tricks

There are some useful variables in the shell that you can use to speed up some tasks.

First of all, after you type in a command, you can hit the up arrow to rerun it (or use the up arrow until you get to the command you want to rerun)

- To do this in a faster way, you can simply type `!!` and press enter. That runs the previously executed command
- Similarly, `!-2` will execute the second most recent command

`!$` is the last argument of the previous command.

For example, 

```bash
grep Hello myfile
cat !$ # will cat myfile
```

# Sources

Here are some very useful sources, and I'd suggest having a look at them for more commands, techniques and ideas.

[Victor Eijkhout's Introduction to HPC Textbook](https://pages.tacc.utexas.edu/~eijkhout/istc/html/unix.html). [PDF](https://tinyurl.com/vle394course)

[The Unix and GNU/Linux command line](https://bootlin.com/doc/legacy/command-line/unix_linux_introduction.pdf)

[Bash Manual](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#Word-Designators)

# Examples and Exercises

There are some examples and a basic file structure in the git repo [here](https://github.com/Michael-Beukman/HPC-InterestGroup).

You can do the following to get started

```bash
git clone https://github.com/Michael-Beukman/HPC-InterestGroup
cd HPC-InterestGroup/shell
```

Then attempt the following exercises

### Simple commands

1. Find all the csv files in the `results` directory
2. Find the last 5 lines of `./results/B/False/2/0.log`
3. What are the column names in `./results/B/False/2/0.csv` ?
4. create the file (and any necessary directories) `results/Z/log1/myfile.txt`

### Searching

1. Find all of the `.csv` files (and the lines in them) in `results` that have a 65 in the first column and an 45 in the last column

### Transforming

1. In the `./results/B/False/1/3.csv` file, find the total of the middle column

### Chaining

1. Find the overall average of the `Accuracy` value across all the `.log` files in the `results` directory

## Potential Answers

There are many possible answers, so if it works, then that's great! Here are just my solutions.

<details>
<summary> Answers: </summary>


1. `find results -name '*.csv'`
2. `tail -n 5 ./results/B/False/2/0.log`
3. `head -n 1 ./results/B/False/2/0.csv`
4. `mkdir -p results/Z/log1 && touch results/Z/log1/myfile.txt`
5. `grep '^65, ' -r results | grep ', 45$'`
6. Two options:
    1. `awk -F',' 'BEGIN {SUM=0} {SUM += $2} END {print SUM}' ./results/B/False/1/3.csv`
    2. `tail -n +1 ./results/B/False/1/3.csv | awk -F',' 'BEGIN {SUM=0} {SUM += $2} END {print SUM}'` (skips the first line)
7. `grep -r 'Final results' results | awk -F'|' '{print $2}' | awk -F': ' 'BEGIN {SUM = 0} {SUM += $2} END {print "Mean Accuracy = " SUM / NR }'`

</details>