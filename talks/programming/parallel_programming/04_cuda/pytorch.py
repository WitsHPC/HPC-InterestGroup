import numpy as np
import torch
from timeit import default_timer as tmr

ROWS   = 10_000
HIDDEN = 10_000
COLS   = 10_000
torch.manual_seed(1)

A = torch.rand((ROWS, HIDDEN))
B = torch.rand((HIDDEN, COLS))

def do_time(A, B, device='cpu', name=None):
    # Runs the matrix multiplication procedure
    if name is None: name = device
    s = tmr()
    A = A.to(device)
    B = B.to(device)
    e = tmr()
    dev_time = e - s
    s = tmr()
    C = A @ B
    e = tmr()
    compute_time = e - s
    total = compute_time + dev_time
    print(f"{name.upper():<10} took {np.round(total, 2):<6}s. {np.round(compute_time, 2):<6}s was spent computing and {np.round(dev_time, 2)}s was spent on moving data.")
    

do_time(A, B, 'cpu')
do_time(A, B, 'cuda')