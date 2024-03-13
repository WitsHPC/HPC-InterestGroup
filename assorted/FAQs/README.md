# Some Questions and Answers
## Symbolic Links
Symbolic Links are ways for you to have the same file referred to by different names.
Say you have a file with the name `test.txt`. You can create a symlink with the command:

```
ln -s test.txt myfile.txt
```
Where the second argument (`myfile.txt`) is arbitrary, it is the name of the symbolic link.
When running `ls -l`, you should see something like:

```
lrwxrwxrwx 1 user user    8 May 23 08:49 myfile.txt -> test.txt
-rw-r--r-- 1 user user 1585 May 23 08:48 test.txt
```

Here, `myfile.txt -> test.txt` means that `myfile` points to `test.txt`.
We can now use `myfile.txt` as just another name for `myfile`. This is useful for two reasons:
1. There is only one file, so if you edit it using one of the names, the underlying file will be edited.
2. It does not copy the file, so you do not need extra disk space if the file is large.

Finally, if we remove `test.txt`, the link turns red, because the file it points to is now gone. When running, e.g. `cat myfile.txt`, we get an error: `cat: myfile.txt: No such file or directory`.

A hard link can avoid this issue. In hard links, if one of the files get deleted, the underlying file will still exist, as long as there is at least one hard link to it.

## OLDPWD not set
This error often happens if you open a new terminal and try to run `cd -`. There is no old pwd set, so it fails to return to where you were before. Just cd'ing to some directories should fix it.
## popd error: directory stack is empty.
This error happens because the `popd` command pops from the directory stack (i.e. the directories you have visited). This is generally finite, so running `popd` enough times will result in the stack being empty and this error happening.
## 2> vs 2>>
The general difference between these is that `2>test.txt` overwrites `test.txt` with the standard error of the command while `2>>test.txt` appends to `test.txt` and does not overwrite what is already there. See [here](https://github.com/WitsHPC/HPC-InterestGroup/tree/main/talks/linux/shell#redirecting-to-files) for more.
## Example of yes
The `yes` command is quite useful, as it allows you to pass input to another program without you typing anything. One use-case of this is when installing programs, or using commands that need input from the user. For instance, `sudo apt-get install vim` asks you for input before it completes. If you are running a script automatically, this will cause problems. In this case, something like `yes | sudo apt-get install vim` will automatically install it, as `yes` inputs `y` to the `apt-get` command.
## AWK: processing is record and fields oriented
AWK is a useful tool, and it processes each line (record) separately, and splits each record into multiple field. You can think of this as a `csv` file, each line is a record and each column's value in that line is a field.
## Octal format for chmod
`chmod` allows us to set permissions for a file or directory. One way to do that is in octal format.
For instance, I can say `chmod 644 file`. This indicates to:
- Give permission `6` to `user`
- Permission `4` to `group`
- Permission `4` to `other`

Each of these numbers can be represented as a binary number of three digits, `rwx`, so
- `111` in binary is `7` in decimal and means that all permissions (read, write and execute) should be given.
- `101` is `5` in decimal, and indicates that only read and execute should be given

And so on.

See [here](https://github.com/WitsHPC/HPC-InterestGroup/tree/main/talks/linux/shell) for more.
## Change Sort Order in Top
While running `top`, you can press several keys to change the sort order. For instance, while in the `top` screen, pressing `M` sorts the processes according to their memory usage.