

migraine:	migraine.o
	gcc $< -L/opt/tensorflow/lib -ltensorflow -o $@

migraine.o:	migraine.c
	gcc -I/opt/tensorflow/include -c $< -o $@


.PHONY:	clean

clean:
	rm -f *.o *.so
