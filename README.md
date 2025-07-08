# Anonymous-OverModRaise

## Build and Run Instructions

```bash
# Compile with debug symbols (note: '-I.' uses a capital 'I', not a lowercase 'l')
g++ -g proposal_main.cpp -o filename -I.

# Launch under gdb
gdb ./filename
run

# If you encounter a stack overflow or segmentation fault due to limited stack size:
ulimit -s unlimited
```

## To reproduce the throughput results in Table 5 of the paper, execute the following six functions in proposal_main.cpp one at a time:

```bash
# 1) logN = 15
evalroundplus_test_S2C_first<15, 28, 30, 7, 7, 49>();
erpluspar_12_S2C_first<15, 29, 30, 7, 7, 10, 20, 21, 49>();
erpluspar_all_together_bootstrap_test_S2C_first<15, 29, 30, 7, 7, 10, 20, 21, 49>();

# 2) logN = 16
evalroundplus_test_S2C_first<16, 37, 42, 4, 5, 58>();
erpluspar_12_S2C_first<16, 38, 42, 4, 5, 16, 26, 30, 58>();
erpluspar_all_together_bootstrap_test_S2C_first<16, 38, 42, 4, 5, 16, 26, 30, 58>();
```


