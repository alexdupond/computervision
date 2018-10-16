/*
  RoVi1
  Example application to load and display an image.


  Version: $$version$$
*/

#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

void show_image(Mat img);
Mat histogram(const Mat& img);
Mat draw_histogram(const Mat& hist);


int main(int argc, char* argv[])
{

  CommandLineParser parser(argc, argv,
          "{help   |            | print this message}"
          "{@imageO | ./pictures/ImageOrig.png | image path}"
          "{@image1 | ./pictures/Image1.png | image path}"
          "{@image2 | ./pictures/Image2.png | image path}"
          "{@image3 | ./pictures/Image3.png | image path}"
          "{@image4_1 | ./pictures/Image4_1.png | image path}"
          "{@image4_2 | ./pictures/Image4_2.png | image path}"
          "{@image5 | ./pictures/Image5_optional.png | image path}"
  );

  if (parser.has("help")){
    parser.printMessage();
    return 0;
  }

  vector<Mat> images;

  // Load image as grayscale
  string filepath = parser.get<string>("@image1");
  Mat img1 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale
  Mat img1Edit = img1;

  filepath = parser.get<string>("@image2");
  Mat img2 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  filepath = parser.get<string>("@image3");
  Mat img3 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  filepath = parser.get<string>("@image4_1");
  Mat img4_1 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  filepath = parser.get<string>("@image4_2");
  Mat img4_2 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  filepath = parser.get<string>("@image5");
  Mat img5 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  filepath = parser.get<string>("@imageO");
  Mat img0 = imread(filepath, IMREAD_GRAYSCALE); // Loads as grayscale

  if(img1.empty()){
    cout << "No image found" << endl;
    return -1;
  }


  namedWindow("og", WINDOW_NORMAL);
  imshow ("og", img0);

  namedWindow("Org", WINDOW_NORMAL);
  imshow ("Org", img2);




  Mat hist = histogram(img1);
//  imshow("Histgram Orgi", draw_histogram(hist));
  hist = histogram(img1Filt);
//  imshow("Histgram Filtered", draw_histogram(hist));

  Mat eqimg;
  equalizeHist(img1, eqimg);
  //imshow("Equalized image", eqimg);

  Mat eqhist = histogram(eqimg);
  //imshow("Equlized histogram", eqhist);


  // Printing the images
  //show_image(img);

  while (cv::waitKey() != 27) // Wait for esc key
      ; // (do nothing)

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
  cout << max << endl;
  Mat img(nbins, nbins, CV_8UC1, Scalar(255));

  for (int i = 0; i < nbins; i++) {
    double h = nbins *(hist.at<float>(i)/max); // Normalizie
    line(img, Point(i, nbins), Point(i, nbins-h), Scalar::all(0));
  }

  return img;
}


/*
  // Filter inspired by https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html
  int kernel_size=30;
  Point anchor = Point( -1, -1 );
  double delta = 0;
  int ddepth =-1;
  Mat img1Filt;
  img1Edit=img2;

  Mat kernel = cv::Mat::ones(kernel_size,kernel_size, CV_32F)/ (float)(kernel_size*kernel_size);
//  cv::filter2D(img1Edit, img1Filt, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
  cv::bilateralFilter(img1Edit, img1Filt, 9, 50, 50);

  namedWindow("Linear filter", WINDOW_NORMAL);
  imshow("Linear filter",img1Filt);

  namedWindow("hist", WINDOW_NORMAL);
  imshow("hist",draw_histogram(histogram(img1Filt)));

  Mat frame;
  cv::GaussianBlur( img1Filt,frame, cvSize(0, 0), 3);

  namedWindow("frame", WINDOW_NORMAL);
  imshow("frame",frame);

  cv::addWeighted(frame, 5 , img1Filt, -0.5, 0, img1Filt);


  namedWindow("Filtered", WINDOW_NORMAL);
  imshow ("Filtered",img1Filt);


*/
