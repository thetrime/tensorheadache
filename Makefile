# You will need LD_LIBRARY_PATH to point to /opt/tensorflow/lib

migraine:	migraine.o mfcc.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -o $@

migraine.o:	migraine.c
	gcc -g -I/opt/tensorflow/include -c $< -o $@

ibuprofen.o:	ibuprofen.c
	gcc -g -I/opt/tensorflow/include -c $< -o $@


mfcc.o:	mfcc.c
	gcc -g -fsanitize=address  -I/opt/fftw/include -c $< -o $@

cluster: cluster.c
	gcc cluster.c -framework AudioToolbox -o cluster

.PHONY:	clean

clean:
	rm -f *.o *.so
