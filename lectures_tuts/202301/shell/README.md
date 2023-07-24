# Basic Linux Shell Commands

Here, if a line starts with a `$`, then it is a command that you can run. Please do not copy/paste the `$` character itself, just whatever comes after.

The other lines indicate sample output when running that command.




You can open a terminal by pressing `ctrl + alt + t`
You can type a command in a terminal, and then press Enter.

## Man Pages
If you are unsure how a command works, you can type `man <command>` and press Enter. This should give you some documentation about what a command does, and how you can use it. You can press `q` to exit.

For instance, `man ls`.

## Navigating the File System


To navigate the file system in Linux, you'll need to know a few basic
commands.

### pwd

The `pwd` command prints the current working directory.

``` {caption="Example usage of the pwd command"}
$ pwd
/home/user
```

**Question for students:** What directory does it print when you run it?

### ls

The `ls` command lists the files and directories in the current working
directory.

``` {caption="Example usage of the ls command"}
$ ls
Desktop Documents Downloads Music Pictures Public Templates Videos
```

**Question for students:** What does it show when you run it?

### cd

The `cd` command is used to change the current working directory.

``` {caption="Example usage of the cd command"}
$ cd Documents/
$ pwd
/home/user/Documents
```

**Question for students:** How do you use it to change the current
working directory to the previous folder?

### mkdir

The `mkdir` command is used to create a new directory.

``` {caption="Example usage of the mkdir command"}
$ mkdir test_directory
$ ls
Desktop Documents Downloads Music Pictures Public Templates Videos test_directory
```

**Question for students:** How do you use it to create a new directory
in a different path?

## Working with Files

Now that we know how to navigate the file system, let's look at some
commands for working with files.

### touch

The `touch` command is used to create an empty file.

``` {caption="Example usage of the touch command"}
$ touch test_file.txt
$ ls
Desktop Documents Downloads Music Pictures Public Templates Videos test_directory test_file.txt
```

**Question for students:** How do you use it to create a bash script?

### cp

The `cp` command is used to copy a file.

``` {caption="Example usage of the cp command"}
$ cp test_file.txt backup_test_file.txt
$ ls
Desktop Documents Downloads Music Pictures Public Templates Videos test_directory test_file.txt backup_test_file.txt
```

**Question for students:** How do you use it to copy a folder?

### mv

The `mv` command is used to move or rename a file.

``` {caption="Example usage of the mv command"}
$ mv test_file.txt new_directory/
$ ls
Desktop Documents Downloads Music Pictures Public Templates Videos test_directory backup_test_file.txt
$ cd new_directory/
$ ls
test_file.txt
$ mv test_file.txt new_name.txt
$ ls
new_name.txt
```

**Question for students:** How do you use it to move or rename a folder?

### rm

The `rm` command is used to delete a file.

``` {caption="Example usage of the rm command"}
$ rm backup_test_file.txt
$ ls
Desktop Documents Downloads Music Pictures Public Templates Videos test_directory
```

**Question for students:** How do you use it to delete a folder and
everything inside?

# Conclusion

In this tutorial, we covered some basic Linux shell commands for
navigating the file system and working with files. These commands are
essential for anyone who wants to use the command line interface in
Linux. With some practice, you'll be able to navigate the file system
and manage files with ease.
