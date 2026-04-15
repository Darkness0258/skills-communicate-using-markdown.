# Virtual GPU Compute Emulator — Build System

CC      = gcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wpedantic \
          -pthread -D_GNU_SOURCE
LDFLAGS = -pthread -lm
TARGET  = vgpu_demo
SRCS    = main.c vgpu.c kernels.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

# Dependency tracking
main.o:    main.c vgpu.h kernels.h
vgpu.o:    vgpu.c vgpu.h
kernels.o: kernels.c kernels.h vgpu.h
