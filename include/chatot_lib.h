#pragma once

#include <string>

enum CLColourFormat
{
    BGR555,
    BGR666,
    BGR888
};

void ChatotLib_Initialize();

void ChatotLib_GetTextFromScreen( void* screenBuffer, unsigned int rows, unsigned int columns, CLColourFormat format, std::string& text);

void ChatotLib_CorrectText( const std::string& inText, std::string& outText );