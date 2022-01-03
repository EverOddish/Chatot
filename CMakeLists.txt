cmake_minimum_required(VERSION 3.10)

project(Chatot)

set(SOURCES src/chatot_lib.cc)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")

add_library(ChatotLib STATIC ${SOURCES})
target_include_directories(ChatotLib PUBLIC include ..\\opencv\\build ..\\opencv\\modules\\core\\include ..\\opencv\\modules\\imgproc\\include ..\\opencv\\modules\\imgcodecs\\include ..\\opencv\\modules\\highgui\\include ..\\opencv\\modules\\videoio\\include ..\\tesseract\\include ..\\tesseract\\build\\include ..\\nuspell\\src)
target_compile_features(ChatotLib PRIVATE cxx_std_17)

#find_package(Tesseract CONFIG REQUIRED)
link_directories(..\\tesseract\\build\\Release)
target_link_libraries(ChatotLib PRIVATE tesseract50)

set(TEST_SOURCES test/main.cc)
add_executable(ChatotTest ${TEST_SOURCES})
target_link_libraries(ChatotTest PRIVATE ChatotLib)
