# You will need LD_LIBRARY_PATH to point to /opt/tensorflow/lib

migraine:	migraine.o mfcc.o
	gcc $^ -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -o $@

migraine.o:	migraine.c
	gcc -I/opt/tensorflow/include -c $< -o $@

mfcc.o:	mfcc.c
	gcc -I/opt/fftw/include -c $< -o $@


.PHONY:	clean

clean:
	rm -f *.o *.so
