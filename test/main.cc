#include <string>
#include <iostream>

#include "chatot_lib.h"

int main(int argc, char** argv)
{
    std::string outText;

    ChatotLib_Initialize();
    ChatotLib_CorrectText("OVAMONDY\n\nCLE\nPRESS START.\n", outText);

    std::cout << outText << std::endl;
}