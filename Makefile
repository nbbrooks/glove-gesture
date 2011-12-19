CXX          =  g++
CFLAGS       = -Wall -g
CXXFLAGS    += `pkg-config opencv --cflags`
LDFLAGS     += `pkg-config opencv --libs`
OBJS =		Gesture.o
TARGET =	gesture

$(TARGET):	$(OBJS) 
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

Gesture.o: Gesture.h Gesture.cpp
	$(CXX) $(CXXFLAGS) -c -o Gesture.o Gesture.cpp

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

