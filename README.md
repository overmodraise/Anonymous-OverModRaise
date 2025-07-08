# Anonymous-OverModRaise

## Build and Run Instructions

To compile and run the code with `gdb`, follow the steps below:

```bash
g++ -g proposal_main.cpp -o filename -I.    # Note: '-I' is a capital 'i', not a lowercase 'L'
gdb ./filename
run
If you encounter a stack overflow or segmentation fault due to limited stack size, set the stack size to unlimited using:


