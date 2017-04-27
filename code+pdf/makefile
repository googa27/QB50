OBJECTS = main.o etat.o quat.o

all: ADCS

ADCS: $(OBJECTS)
	g++ -g -o $@ $^

%.o: %.cpp
	clang -g -o $@ $^

clean:
	rm $(OBJECTS) adcs
