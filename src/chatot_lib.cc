#include <iostream>
#include <vector>
#include <cmath>

#include "chatot_lib.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

int thresh = 50, N = 10;

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

void ChatotLib_GetTextFromScreen(void* screenBuffer, unsigned int rows, unsigned int columns, CLColourFormat format, std::string& text)
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
                cvType = CV_8UC3;
                break;
        }

        Mat image((int)rows, (int)columns, cvType, screenBuffer);
        vector<vector<Point>> squares;

        findSquares(image, squares);
        cout << "Found " << squares.size() << " squares" << endl;
    }
    catch (cv::Exception& e)
    {
        cout << e.what() << endl;
    }
}