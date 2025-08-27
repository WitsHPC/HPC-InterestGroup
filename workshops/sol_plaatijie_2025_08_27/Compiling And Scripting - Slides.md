# Compiling Software

## Why ?

---
- Architecture differences
- Hardware utilisation
- Performance
--- 

## How?

- Build tools
- Read the instructions!
---

# HWLOC

---

```bash
wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.2.tar.bz2
```

--- 

```bash
man tar
```

```bash
tar -xvf hwloc-2.12.2.tar.bz2
```

```bash
cd hwloc-2.12.2
```

---
![[attachments/lstopo output.png]]

---

## Configuring

```bash
./configure -h
```

---

```bash
export CFLAGS="-O3 -march=native"
export CXXFLAGS="-O3 -march=native"
mkdir build
./configure --prefix=$HOME/hwloc-2.12.2/build
```

---

## MAking

```bash
make -j $(nproc)
```

```bash
make check
```

```bash
make install
```

---

# Scripting

```bash
rm -r hwloc-2.12
rm hwloc-2.12.2.tar.bz2
```

> Now put everything we did before into a script and use `bash <script_name>.sh` to download, configure, make and run