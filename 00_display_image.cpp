/*
  RoVi1
  Example application to load and display an image.


  Version: $$version$$
*/

#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void show_image(Mat img);
Mat histogram(const Mat& img);
Mat draw_histogram(const Mat& hist);

int main(int argc, char* argv[])
{

  CommandLineParser parser(argc, argv,
          "{help   |            | print this message}"
          "{@image | ./lena.bmp | image path}"
  );

  if (parser.has("help")){
    parser.printMessage();
    return 0;
  }

      // Load image as grayscale
  string filepath = parser.get<string>("@image");
  Mat img = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  if(img.empty()){
    cout << "No image found" << endl;
    return -1;
  }

  Mat hist = histogram(img);
  //imshow("Histgram", draw_histogram(hist));

  Mat eqimg;
  equalizeHist(img, eqimg);
  imshow("Equalized image", eqimg);

  Mat eqhist = histogram(eqimg);
  imshow("Equlized histogram", eqhist);

  // Printing the images
  //show_image(img);

  waitKey(0);
  return 0;
}






void show_image(Mat img){
  namedWindow("Display window", WINDOW_NORMAL);
  imshow("Display window", img);
}

Mat histogram(const Mat& img){

  assert(img.type() == CV_8UC1);

  Mat histogram;

  calcHist(
    vector<Mat>{img},
    {0}, // Channels
    noArray(), //mask
    histogram, // Output histogram
    {256}, // Hisogram size / Number of bins
    {0, 256} // Pairs of bin lower and upper
  );
  return histogram;
}

Mat draw_histogram(const Mat& hist){
  int nbins = hist.rows;
  double max = 0;
  minMaxLoc(hist, nullptr, &max);
  Mat img(nbins, nbins, CV_8UC1, Scalar(255));

  for (int i = 0; i < nbins; i++) {
    double h = nbins *(hist.at<float>(i)/max); // Normalizie
    line(img, Point(i, nbins), Point(i, nbins-h), Scalar::all(0));
  }

  return img;
}
