C_OPTIONS=-O6
#C_OPTIONS=  -g -Wall
CC = gcc


SOURCES = display.o patternFileIO.o learning.o unlearning.o pseudolearning.o sampling.o allocation.o general.o random.o net.o training.o param.o 

#a: all
#	for f in $(SOURCES)
#		echo $f
#	done


all: netCode

#random.o net.o training.o allocation.o learning.o unlearning.o pseudolearning.o sampling.o display.o general.o param.o annescode.c
#display.o learning.o unlearning.o pseudolearning.o sampling.o allocation.o general.o random.o net.o training.o param.o 

netCode: $(SOURCES) annescode.c 
	$(CC) -lm $(C_OPTIONS) -o net $(SOURCES) annescode.c 
# /usr/lib/libMallocDebug.a

net.o: random.o general.o net.c net.h
	$(CC) -c $(C_OPTIONS) net.c

training.o: random.o general.o net.o training.h training.c
	$(CC) -c $(C_OPTIONS) training.c

general.o: random.o general.c general.h
	$(CC) -c $(C_OPTIONS) general.c

allocation.o: allocation.c allocation.h net.o training.o
	$(CC) -c $(C_OPTIONS) allocation.c

sampling.o: random.o general.o net.o training.o sampling.c sampling.h
	$(CC) -c $(C_OPTIONS) sampling.c

display.o: random.o general.o net.o training.o display.c display.h
	$(CC) -c $(C_OPTIONS) display.c

patternFileIO.o: general.o net.o training.o patternFileIO.c patternFileIO.h
	$(CC) -c $(C_OPTIONS) patternFileIO.c

learning.o: general.o net.o training.o learning.c learning.h
	$(CC) -c $(C_OPTIONS) learning.c

pseudolearning.o: general.o net.o training.o pseudolearning.c pseudolearning.h
	$(CC) -c $(C_OPTIONS) pseudolearning.c

unlearning.o: general.o net.o training.o unlearning.c unlearning.h
	$(CC) -c $(C_OPTIONS) unlearning.c

random.o: random.c random.h
	$(CC) -c $(C_OPTIONS) random.c

param.o: param.c param.h
	$(CC) -c $(C_OPTIONS) param.c


clean:
	rm net; rm -f *.o