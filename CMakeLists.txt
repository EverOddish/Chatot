cmake_minimum_required(VERSION 3.10)

project(Chatot)

set(SOURCES src/chatot_lib.cc)

add_library(ChatotLib STATIC ${SOURCES})
target_include_directories(ChatotLib PUBLIC include ..\\opencv\\build ..\\opencv\\modules\\core\\include ..\\opencv\\modules\\imgproc\\include ..\\opencv\\modules\\imgcodecs\\include ..\\opencv\\modules\\highgui\\include ..\\opencv\\modules\\videoio\\include)
