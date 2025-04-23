# cmatrix

A simple and fun terminal-based "Matrix rain" animation written in C.

This README shows you how to **install cmatrix from source** using `CMake` on a typical Ubuntu system.

---

## 🧰 Prerequisites

Make sure you have the following packages installed:

Note: this step is not required if you are using the MS Lab Computers

```bash
sudo apt update
sudo apt install cmake build-essential libncurses5-dev
```


## Build Instructions

Clone the repository:


```bash
git clone https://github.com/abishekvashok/cmatrix.git
cd cmatrix
```

Create a build directory and move into it:

```bash
mkdir build && cd build
```

Generate the build files with CMake:

```bash
cmake ..
```

Compile the source:

```bash
make
```

## Run cmatrix

After a successful build, you can run the program directly:

```bash
./cmatrix
```

## Optional Flags

You can customize the behavior of cmatrix with various flags. Some examples:

-a: Asynchronous scroll

-b: Bold characters

-u [delay]: Set scroll delay (e.g. -u 3)

-s: "Screensaver" mode (exit with Ctrl+C)

```bash
cmatrix -ab
```
