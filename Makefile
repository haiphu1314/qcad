# Variable to specify the compiler
CC = gcc -msse4.2

# Compilation flags
CFLAGS = -Wall -Wextra -fopenmp -I./src -mavx2 

# Linker flags for BLAS
LDFLAGS = -fopenmp

# Directory containing the source files
SRC_DIR = src

# Main source file
MAIN = test_qcad.c
# MAIN = test_lenet.c
# MAIN = test_alex.c
# MAIN = test_vgg11.c


# Other source files
SRCS = $(SRC_DIR)/conv.c $(SRC_DIR)/linear.c $(SRC_DIR)/model.c $(SRC_DIR)/utils.c

# Object files generated from the source files
OBJS = $(MAIN:.c=.o) $(SRCS:.c=.o)

# Name of the executable file
TARGET = main

# Default build rule
all: $(TARGET)
	make run
	make clean

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Rule to run the program
run: $(TARGET)
	./$(TARGET)

bug: $(TARGET)
	gdb ./$(TARGET)

.PHONY: all clean run
