# Anonymous-OverModRaise

# Build and Run Instructions
g++ -g proposal_main.cpp -o filename -I.    # Note: '-I' is a capital 'i', not a lowercase 'L'
gdb ./filename
run

# If you encounter a stack overflow or segmentation fault due to limited stack size, set the stack size to unlimited:
ulimit -s unlimited

# The following six functions can be executed one at a time in proposal_main.cpp 
# to reproduce the Throughput results in Table 5 of the paper:

# (1) logN = 15
evalroundplus_test_S2C_first<15, 28, 30, 7, 7, 49>();
erpluspar_12_S2C_first<15, 29, 30, 7, 7, 10, 20, 21, 49>();
erpluspar_all_together_bootstrap_test_S2C_first<15, 29, 30, 7, 7, 10, 20, 21, 49>();

# (2) logN = 16
evalroundplus_test_S2C_first<16, 37, 42, 4, 5, 58>();
erpluspar_12_S2C_first<16, 38, 42, 4, 5, 16, 26, 30, 58>();
erpluspar_all_together_bootstrap_test_S2C_first<16, 38, 42, 4, 5, 16, 26, 30, 58>();
