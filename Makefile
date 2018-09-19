# You will need LD_LIBRARY_PATH to point to /opt/tensorflow/lib

CFLAGS=-I/opt/fftw/include -I/opt/tensorflow/include
SWI_CFLAGS=`PKG_CONFIG_PATH=/opt/swi-prolog/lib/pkgconfig/ pkg-config swipl --cflags`
SWI_LDFLAGS=`PKG_CONFIG_PATH=/opt/swi-prolog/lib/pkgconfig/ pkg-config swipl --libs`
migraine:	migraine.o mfcc.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -o $@

migraine2d:	migraine.o mfcc2d.o ibuprofen2d.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -o $@

holmes.o: holmes.c
	gcc -g $(CFLAGS) -c $< -o $@

migraine.o:	migraine.c
	gcc -g $(CFLAGS) -c $< -o $@

ibuprofen.o:	ibuprofen.c
	gcc -g $(CFLAGS) -c $< -o $@

ibuprofen2d.o:	ibuprofen2d.c
	gcc -g $(CFLAGS) -c $< -o $@

mfcc.o:	mfcc.c
	gcc -g -fsanitize=address $(CFLAGS) -c $< -o $@

mfcc2d.o:	mfcc2d.c
	gcc -g -fsanitize=address $(CFLAGS) -c $< -o $@

elementary: holmes.o ibuprofen.o migraine.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -o $@

cluster: cluster.o mfcc.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -framework AudioToolbox -o $@

cluster.o: cluster.c
	gcc -c cluster.c $(CFLAGS) -o $@

baker: baker.o holmes.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -framework AudioToolbox -o $@

baker.o: baker.c block.c
	gcc -c baker.c $(CFLAGS) -o $@

bakerloo: bakerloo.o holmes.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 -framework AudioToolbox -o $@

bakerloo.o: bakerloo.c block.c
	gcc -c bakerloo.c $(CFLAGS) -o $@


paracetamol: ibuprofen.o mfcc.o paracetamol.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3  -o $@

paracetamol.o: paracetamol.c
	gcc -c paracetamol.c $(CFLAGS) -o $@


.PHONY:	clean

clean:
	rm -f *.o *.so

libuprofen.so: libuprofen.o holmes.o ibuprofen.o
	gcc $^ -fsanitize=address -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 $(SWI_LDFLAGS) -shared -o $@

libuprofen.o: libuprofen.c
	gcc -c libuprofen.c $(SWI_CFLAGS) $(CFLAGS) -o $@

wavy.o: wavy.c
	gcc -c wavy.c -o wavy.o

libuprofen-wavy.so: libuprofen.o holmes.o wavy.o ibuprofen.o
	gcc $^ -L/opt/tensorflow/lib -ltensorflow -L/opt/fftw/lib -lfftw3 $(SWI_LDFLAGS) -shared -o $@
