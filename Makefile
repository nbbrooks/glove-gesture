CXX          =  g++
CFLAGS       = -Wall -g
CXXFLAGS    += `pkg-config opencv --cflags`
LDFLAGS     += `pkg-config opencv --libs`
OBJS =		Gesture.o LargePrint.o
TARGET =	gesture

$(TARGET):	$(OBJS) 
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

Gesture.o: Gesture.h Gesture.cpp
	$(CXX) $(CXXFLAGS) -c -o Gesture.o Gesture.cpp

LargePrint.o: LargePrint.h LargePrint.cpp
	$(CXX) $(CXXFLAGS) -c -o LargePrint.o LargePrint.cpp

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

