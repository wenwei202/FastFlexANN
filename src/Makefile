IDIR =../inc
CC = g++
AR = ar
CFLAGS = -g -fpermissive -I$(IDIR)

TGT = main

OBJS =  common.o \
ff_data_set.o \
ff_nn.o \
ff_sw.o \
main.o \

build = \
	@if [ -z "$V" ]; then \
		echo '	[$1]	$@'; \
		$2; \
	else \
		echo '$2'; \
		$2; \
	fi

% : %.o
	$(call build,LINK,$(CC) $(CFLAGS) $(OBJS)  -o $@ $(LFLAGS))

%.o : %.cpp 
	$(call build,CC,$(CC) $(CFLAGS) -c $< -o $@)

%.a : %.o
	$(call build,AR,$(AR) rcs $@ $^)


all: $(OBJS) $(TGT)


clean: 
	rm -f *.o $(OBJS) $(TGT)
