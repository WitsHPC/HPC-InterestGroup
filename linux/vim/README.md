# Vim

## Intro
Today we will be going over [Vim](https://www.vim.org/), a useful and powerful command line text editor.

+ Through a unique approach, it can allow you to edit files much faster
+ This does come at a price of not being familiar to users of standard text editors
+ This talk has a few main goals:
	+ Explain what Vim is and why you would want to use it
	+ Show you enough to be able to use Vim normally
	+ Make you interested in learning more by showing some cool things.

---

## Why Vim?
+ You may ask why we would need to use command line editor at all, why not just use an IDE, notepad, or Visual Studio Code?
+ One reason is convenience: 
	+ If you are in the terminal and need to edit a file, it takes much longer to navigate to that folder, open it in another editor, etc. than it does just using a terminal editor
+ The other is necessity
	+ On many HPC systems, we only have terminal, non-graphical access. So, when you need to edit a file, the terminal is the way to go. VsCode's ssh remote extension is useful, but not really for by-the-way file edits.
+ Finally, power
	+ Vim power users can truly do astonishing things. If you get good enough, and know enough, you can be very productive using Vim.

---

## Idiosyncrasies
+ Vim is more designed to focus on editing text, rather than writing it out. This is generally something we do very often, so it does make sense.
+ Vim has multiple *modes*, in contrast to a standard editor.
	+ Insert mode: Where typing keys puts those characters in the file
	+ Command mode: Where each key/combination has a special function
---

## Basic Usage
+ To use Vim, you can type `vim filename` into an terminal, and a window should appear 
	+ (`sudo apt install vim` first or use `vi` if this returns a not-found error)
+ To exit Vim
	+ Press `Escape`
	+ Type `:q!`
	+ Press Enter
+ To save the file:
	+ Press `Escape`
	+ Type `:w`
	+ Press Enter
---

## Modes
+ Insert Mode:
	+ Press `i` to get here
	+ This allows you to type characters like normal
+ Command Mode:
	+ Press `Escape` to get here
	+ This allows you to enter commands to do things.
---
## Motions
+ One key part of Vim is motions: In command mode, how do you move around. The following are common and useful:
	+ `h,j,k,l` Move the cursor Left, Down, Up, Right respectively.
	+ `0`: Move to the beginning of the line
	+ `$`: End of the line
	+ `gg`: Beginning of the file
	+ `G`: End of the file
	+ `w`: Move to the start of the next work
	+ `b`: Move to the end of the next work
+ An additional feature is: to repeat a motion multiple times, prefix it with a number. So, `10l` will move 10 characters to the right.
---
## Editing
+ Now, in command mode, the following *edits* are useful
	+ `x`: Delete a character
	+ `r`: Replace the current character with a new one. For instance, `rc` will replace the current character with `c`
	+ `u` Undo the last change
	+ `ctrl+r`: Redo the last undo
+ And remember, `i` puts you in insert mode to type normally.
---
## Editing
+ `I` puts you in insert mode at the beginning of the line
+ `A`: Insert mode at the end of the line
+ `o`: Creates a new line below the cursor
+ `O` Creates a new line above the cursor
---
## Editing
+ One very powerful feature of Vim is the ability to combine *motions* and *edits*
+ `d` **D**eletes, so `dl` will delete the character to the right. 
	+ `d10l` will delete the 10 characters to the right
+ `c` **C**hanges (deletes and puts you in insert mode).
	+ `c0` will change from the current location to the start of the line.
+ `y` **Y**anks (Copies)
+ Any of the above, if you press the key twice, it works on an entire line (`dd` deletes an entire line, `yy` copies it)
+ If you use the capital version, it does it until the end of the line
+ `p` **P**uts (pastes)

---
## More Motions
+ `t<char>` goes **till** the next instance of `<char>`, i.e. to the character just before it.
	+ `T<char>` goes till the previous instance
+ `f<char>` **finds** the next instance of `<char>`, landing the cursor on this.
	+ Use like `fg` -> Goes to the first `g`
	+ `F<char>` finds the previous instance.
+ `iw`: To be used in conjunction with `d`, `c` or `y`: Does this operation **inside the current word**
	+ `aw` Does the operation **around** the current word, i.e. includes the spaces/punctuation.
+ Can use this with other items too!
	+ `yi{`: Copies everything inside the current curly-braces
	+ `ca)`: Changes everything around the current set of round brackets (Including the brackets themselves, use `ci)` to not include the brackets)
	+ `di"`: Deletes everything in the current set of quotation marks.
---

## Searching
+ You can search by typing `/<thing to search for>` and pressing Enter
	+ Use `n` to go to the next hit
	+ Use `N` to go to the previous
+ `:s/<search>/<replace>/g`: Replaces text in the current line
+`:%s/<search>/<replace>/g`: Replaces text in the current file
---
## Visual Mode
+ Visual Mode is a way to select text, and then do something with it.
	+ Press `v` to enter visual mode, and use motions to select text
	+ Then e.g. `Y`ank it, etc.
+ Use `V` to enter visual mode until the end of the line
+ Use `ctrl+v` to enter Visual block mode, where you can select individual characters per line.
---
## Period and Macros
- You can use `.` to repeat the previous command
- `ctrl xf`: Autocompletes filenames
- `ctrl xn`: Autocompletes names of items in the current file
- You can use macros to automate tasks
	- `q<name><commands><Escape>q`
	- `<name>` is a letter from `a` to `z`
	- `<commands>` is a string of commands
- Call it as `@<name>`
---
## Even More
+ What we covered here is just a small fraction of all that Vim has to offer.
+ There are lots more, including but not limited to
	+ Registers
	+ More Macros
	+ Different sessions
+ Execute terminal inside the editor
	+ `:! <bash command>`
+ Configurations, `vimrc`

---
## Conclusion
+ Hopefully this is a nice introduction to Vim, and it motivates you to learn more, and to start using it!
	+ Start small, and gradually increase your usage as you improve.

---


## Resources
- Use `vimtutor` in your terminal
- https://linuxfoundation.org/blog/classic-sysadmin-vim-101-a-beginners-guide-to-vim/
- https://www.quora.com/What-are-the-most-amazing-things-that-can-be-done-with-Vim
- https://www.freecodecamp.org/news/learn-linux-vim-basic-features-19134461ab85/
- https://www.twilio.com/blog/5-quality-of-life-vim-tricks-for-your-vimrc
- https://levelup.gitconnected.com/7-surprising-vim-tricks-that-will-save-you-hours-b158d23fe9b7
- https://www.redhat.com/sysadmin/use-vim-macros
- https://www.redhat.com/sysadmin/use-vim-macros