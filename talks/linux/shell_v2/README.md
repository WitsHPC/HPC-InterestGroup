# Shell V2 & Bash Programming

- [Shell V2 & Bash Programming](#shell-v2--bash-programming)
- [Intro](#intro)
- [All of the things](#all-of-the-things)
  - [SSH Configs](#ssh-configs)
  - [Tmux](#tmux)
- [Bash](#bash)
  - [Special commands / tokens](#special-commands--tokens)
  - [For](#for)
  - [If](#if)
  - [Functions](#functions)
  - [Arguments and bash scripts](#arguments-and-bash-scripts)
  - [Error Checking](#error-checking)
- [Python](#python)
- [Easy parallelisation](#easy-parallelisation)
- [Summary](#summary)
- [Sources](#sources)
# Intro

This is somewhat of a sequel to the first [Shell Talk](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/shell), that won't necessarily cover individual commands but rather some more holistic components, in particular we will cover some bash programming concepts at the end that could be very useful.

# All of the things

## SSH Configs

The TLDR here is:

edit `~/.ssh/config` and do something like this:

```bash
Host my_server1
     HostName 123.456.17.8
     User myname

 Host my_server2
     HostName 10.0.0.2
     User myname2
     ProxyJump my_server1
```

Then, instead of typing `ssh myname@123.456.17.8`, you can simply type `ssh my_server1`.

Similarly, `ssh myname2@10.0.0.2 -J myname@123.456.17.8` can be replaced with `ssh my_server2`.

Be sure to combine this with SSH Keys, so you can ssh without having to type your password all the time.

## Tmux

The notes from the previous Shell talk can be found [here](https://github.com/Michael-Beukman/HPC-InterestGroup/tree/main/shell#tmux), this will just be a simple demo.

# Bash

Bash is pretty cool, and it has lots of 'standard' programming constructs

## Special commands / tokens

## For

There are a few different ways to do for loops in bash, but they all use basically the same concept: iterating over whitespace-separated words.

```bash
for i in 1 2 3 4 5; do
     echo $i
 done
```

But, we can also create a range

```bash
# inclusive
for i in {1..5}; do
     echo $i
 done
```

We could also use the output of other commands as things to iterate over, like `ls` (or anything else → `cat`, `grep`, etc.)

```bash
for i in `ls *`; do
    echo $i
done
```

There are also while and until loops, see [here](https://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO-7.html#ss7.1).

## If

Ifs are relatively simple, like:

```bash
A=1
B=2

if [ "$A" = "$B" ]; then
    echo "$A = $B is True!"
fi
```

We could also have else ifs and other conditions:

```bash
# use -lt and -gt for less than and greater than respectively
if [ "$A" -lt "$B" ]; then
    echo "$A < $B is True!"
elif ["$A" -gt "$B"]; then
    echo "$A > $B is True!"
else
    echo "$A = $B is True"
fi
```

## Functions

Functions can take arguments, which are referred to as `$1`, `$2`, `$3` and so on.

```bash
function greet {
    echo "Hello $1"
}

greet John
greet "John Smith"
```

## Arguments and bash scripts

Bash scripts themselves can take arguments, again simply using the positional structure for functions.

There are also a few special variables, like

- `$@` → All parameters
- `$#` → The number of parameters

```bash
echo "We got $# arguments"

echo "They were: $@
echo "The first argument was $1, second was $2, third was $3"
```

`./script.sh alpha beta gamma` yields

```bash
We got 3 arguments
They were:  alpha beta gamma
The first argument was alpha, second was beta, third was gamma
```

You can also do named arguments, e.g. `./abc.sh --param1 x --param2 y`, but that feels a bit too convoluted to be very useful for small scripts. [This](http://wiki.bash-hackers.org/howto/getopts_tutorial) tutorial might help there, but if you really need this functionality, I'd suggest just using python.

## Error Checking

Sometimes you want to check the results of operations, and print a message / exit if something failed.

This is often done if the following commands depend on the previous ones succeeding. A pretty common pattern is:

`command 2>/dev/null || echo "ERROR" && exit 1`

The `2>/dev/null` redirects stderror to `/dev/null`, which in effect mutes it. You can remove that if you want to see the actual error that caused the failure.

```bash
cat /tmp/this_file_does_not_exist 2>/dev/null || echo "The file does not exist. Exiting now" && exit 1
touch /tmp/this_file_does_not_exist/some/dir/file  2>/dev/null || echo "Some problem occurred. Exiting now" && exit 1
echo "The above is a result of an operation"
```

# Python

If you prefer, you can mostly write your bash scripts in python  ;)

Like this ([source](https://geekflare.com/python-run-bash/))

```bash
import subprocess
result = subprocess.run(["ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('STDOUT:', result.stdout.decode('utf-8'))
print('STDERR:', result.stderr.decode('utf-8'))
```

The main idea is to use `subprocess.run(cmd)`

# Easy parallelisation

If you have a long running command, and you need to run multiple independent copies of them, then using the following can be quite useful, and super simple:

The main concept is to use `&` to run in the background, and use `wait` to wait for completion.

```bash
for i in {1..5}; do
    echo "Starting $i" && sleep 5 && echo "Ending $i" &
done
wait
echo "DONE"
```

# Summary

Here we covered some bash scripting concepts, and introduced some other useful things like tmux and ssh configs. The scripts and notes will be uploaded to: [https://github.com/Michael-Beukman/HPC-InterestGroup/](https://github.com/Michael-Beukman/HPC-InterestGroup/)

# Sources

Cool tutorial: [https://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO.html](https://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO.html)

SSH Config: [https://linuxize.com/post/using-the-ssh-config-file/](https://linuxize.com/post/using-the-ssh-config-file/)

Python bash scripting: [https://geekflare.com/python-run-bash/](https://geekflare.com/python-run-bash/)
