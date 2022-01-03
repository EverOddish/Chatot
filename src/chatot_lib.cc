#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

#include "chatot_lib.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <tesseract/baseapi.h>

#include <nuspell/dictionary.hxx>
#include <nuspell/finder.hxx>

using namespace cv;
using namespace std;

int thresh = 50, N = 10;

static tesseract::TessBaseAPI* g_api = NULL;
static nuspell::v5::Dictionary g_dict;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

static bool squaresDifferent(vector<Point> square1, vector<Point> square2)
{
    int threshold = 10;

    // Calculate the distance between the bottom left points of each square
    int dx1 = abs(square1[0].x - square2[0].x);
    if (dx1 <= threshold)
    {
        int dy1 = abs(square1[0].y - square2[0].y);
        if (dy1 <= threshold)
        {
            // Calculate the distance between the top right points of each square
            int dx2 = abs(square1[2].x - square2[2].x);
            if (dx2 <= threshold)
            {
                int dy2 = abs(square1[2].y - square2[2].y);
                if (dy2 <= threshold)
                {
                    // Squares are more or less the same
                    return false;
                }
            }
        }
    }

    // Squares are different
    return true;
}

static void deduplicateSquares(vector<vector<Point>>& squares)
{
    vector<vector<Point>> current = squares;
    vector<vector<Point>> keep;
    vector<Point> squareToCompare;
    bool changed = true;

    if (0 == squares.size())
    {
        return;
    }

    //cout << "Number of squares before: " << squares.size() << endl;

    while (changed)
    {
        keep.clear();
        changed = false;

        squareToCompare = current[0];

        for (vector<Point> otherSquare : current)
        {
            // Don't compare with self
            if (squareToCompare != otherSquare)
            {
                if (squaresDifferent(squareToCompare, otherSquare))
                {
                    keep.push_back(otherSquare);
                }
            }
        }
        keep.push_back(squareToCompare);

        if (current.size() != keep.size())
        {
            changed = true;
            current = keep;
        }
    }

    squares = current;
    //cout << "Number of squares after: " << squares.size() << endl;
}

// returns sequence of squares detected on the image.
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    cv::cvtColor(timg, gray0, cv::COLOR_BGR2GRAY);

    // try several threshold levels
    for (int l = 0; l < N; l++)
    {
        // hack: use Canny instead of zero threshold level.
        // Canny helps to catch squares with gradient shading
        if (l == 0)
        {
            // apply Canny. Take the upper threshold from slider
            // and set the lower to 0 (which forces edges merging)
            Canny(gray0, gray, 0, thresh, 5);
            // dilate canny output to remove potential
            // holes between edge segments
            dilate(gray, gray, Mat(), Point(-1, -1));
        }
        else
        {
            // apply threshold if l!=0:
            //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
            gray = gray0 >= (l + 1) * 255 / N;
        }

        // find contours and store them all as a list
        findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        vector<Point> approx;

        // test each contour
        for (size_t i = 0; i < contours.size(); i++)
        {
            // approximate contour with accuracy proportional
            // to the contour perimeter
            approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

            // square contours should have 4 vertices after approximation
            // relatively large area (to filter out noisy contours)
            // and be convex.
            // Note: absolute value of an area is used because
            // area may be positive or negative - in accordance with the
            // contour orientation
            if (approx.size() == 4 &&
                fabs(contourArea(approx)) > 1000 &&
                isContourConvex(approx))
            {
                double maxCosine = 0;

                for (int j = 2; j < 5; j++)
                {
                    // find the maximum cosine of the angle between joint edges
                    double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                    maxCosine = MAX(maxCosine, cosine);
                }

                // if cosines of all angles are small
                // (all angles are ~90 degree) then write quandrange
                // vertices to resultant sequence
                if (maxCosine < 0.3)
                    squares.push_back(approx);
            }
        }
    }

    deduplicateSquares(squares);
}

int lowestX(vector<Point>& square)
{
    int lowest = 1000000;

    for (Point p : square)
    {
        if (p.x < lowest)
        {
            lowest = p.x;
        }
    }

    return lowest;
}

int highestX(vector<Point>& square)
{
    int highest = 0;

    for (Point p : square)
    {
        if (p.x > highest)
        {
            highest = p.x;
        }
    }

    return highest;
}

int lowestY(vector<Point>& square)
{
    int lowest = 1000000;

    for (Point p : square)
    {
        if (p.y < lowest)
        {
            lowest = p.y;
        }
    }

    return lowest;
}

int highestY(vector<Point>& square)
{
    int highest = 0;

    for (Point p : square)
    {
        if (p.y > highest)
        {
            highest = p.y;
        }
    }

    return highest;
}

void ChatotLib_Initialize()
{
    // Initialize Tesseract
    g_api = new tesseract::TessBaseAPI();
    g_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    g_api->SetPageSegMode(tesseract::PSM_AUTO);
    g_api->SetVariable("debug_file", "NUL");

    // Initialize nuspell
    g_dict = nuspell::Dictionary::load_from_path("en_US");
}

void ChatotLib_GetTextFromScreen(void* screenBuffer, unsigned int rows, unsigned int columns, CLColourFormat format, std::string& text)
{
    if (screenBuffer && rows > 0 && columns > 0)
    {
        try
        {
            int cvType = 0;

            switch (format)
            {
            case BGR555:
                break;
            case BGR666:
                break;
            case BGR888:
                cvType = CV_8UC4;
                break;
            }

            //cout << "image rows=" << rows << " columns=" << columns << endl;
            Mat image((int)columns, (int)rows, cvType, screenBuffer);

            //Mat image2;
            //cvtColor(image, image2, COLOR_RGBA2BGR);

            // Used for dumping raw RGB data
            //ofstream myfile;
            //myfile.open("C:\\temp\\dump.bin");
            //myfile.write((const char*)screenBuffer, rows * columns * 4);
            //myfile.close();

            //imwrite("C:\\temp\\dump.bmp", image);

            vector<vector<Point>> squares;

            findSquares(image, squares);
            //cout << "Found " << squares.size() << " squares" << endl;

            for (vector<Point> square : squares)
            {
                // TODO: Enlarge square slightly to make sure text isn't cut off

                //cout << "0: (" << square[0].x << "," << square[0].y << ")" << endl;
                //cout << "1: (" << square[1].x << "," << square[1].y << ")" << endl;
                //cout << "2: (" << square[2].x << "," << square[2].y << ")" << endl;
                //cout << "3: (" << square[3].x << "," << square[3].y << ")" << endl;

                int lowestXarg = lowestX(square);
                int highestXarg = highestX(square);
                int lowestYarg = lowestY(square);
                int highestYarg = highestY(square);
                //cout << "lowestXarg=" << lowestXarg << " highestXarg=" << highestXarg << endl;
                //cout << "lowestYarg=" << lowestYarg << " highestYarg=" << highestYarg << endl;

                Range rowsRange(lowestYarg, highestYarg);
                Range colsRange(lowestXarg, highestXarg);
                Mat cropped(image, rowsRange, colsRange);
                //imwrite("C:\\temp\\dump.bmp", cropped);

                if (g_api)
                {
                    //cout << "cols=" << cropped.cols << " rows=" << cropped.rows << " step=" << cropped.step << endl;
                    g_api->SetImage(cropped.data, cropped.cols, cropped.rows, 4, cropped.step);
                    char* text_ptr = g_api->GetUTF8Text();
                    if (text_ptr)
                    {
                        text = string(text_ptr);
                    }
                }
            }
        }
        catch (cv::Exception& e)
        {
            cout << e.what() << endl;
        }
    }
}

void ChatotLib_CorrectText(const std::string& inText, std::string& outText)
{
    string inTextCopy(inText);
    string space_delimiter = " ";
    vector<string> words{};
    vector<string> sugs{};

    std::replace(inTextCopy.begin(), inTextCopy.end(), '\n', ' ');

    size_t pos = 0;
    while ((pos = inTextCopy.find(space_delimiter)) != string::npos) {
        words.push_back(inTextCopy.substr(0, pos));
        inTextCopy.erase(0, pos + space_delimiter.length());
    }

    size_t i = 0;
    for (auto iter = words.begin(); iter != words.end(); iter++)
    {
        auto word = *iter;
        if (g_dict.spell(word))
        {
            outText += word + " ";
        }
        else
        {
            sugs.clear();
            g_dict.suggest(word, sugs);

            if (!sugs.empty())
            {
                outText += sugs[0] + " ";
            }
            else
            {
                outText += word + " ";
            }
        }
        i++;
    }
}