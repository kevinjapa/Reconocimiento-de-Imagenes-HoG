all:
	g++ Principal.cpp --std=c++17 -I/usr/local/include/opencv4/ -L/Users/kevinjapa/opencv/build/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_ml -lopencv_features2d -lopencv_objdetect -o vision.bin -Wl,-rpath,/Users/kevinjapa/opencv/build/lib/

saludo:
	echo "Hola C++"

run:
	./vision.bin
