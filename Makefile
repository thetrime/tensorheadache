# You will need LD_LIBRARY_PATH to point to /opt/tensorflow/lib

CFLAGS=-I/opt/fftw/include -I/opt/tensorflow/include

migraine:	migraine.o mfcc.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -o $@

migraine.o:	migraine.c
	gcc -g $(CFLAGS) -c $< -o $@

ibuprofen.o:	ibuprofen.c
	gcc -g $(CFLAGS) -c $< -o $@


mfcc.o:	mfcc.c
	gcc -g -fsanitize=address $(CFLAGS) -c $< -o $@

cluster: cluster.o mfcc.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -framework AudioToolbox -o $@

cluster.o: cluster.c
	gcc -c cluster.c $(CFLAGS) -o $@

paracetamol: ibuprofen.o mfcc.o paracetamol.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3  -o $@

paracetamol.o: paracetamol.c
	gcc -c paracetamol.c $(CFLAGS) -o $@


.PHONY:	clean

clean:
	rm -f *.o *.so
