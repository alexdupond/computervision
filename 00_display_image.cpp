/*
  RoVi1
  Example application to load and display an image.


  Version: $$version$$
*/

#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;

void drawRegtangle(cv::Mat img, cv::Vec3b color);

void rectang(cv::Mat image, cv::Vec3b color);

void grayscale(cv::Mat image);

int main(int argc, char* argv[])
{
    // Parse command line arguments -- the first positional argument expects an
    // image path (the default is ./book_cover.jpg)
    cv::CommandLineParser parser(argc, argv,
        // name  | default value    | help message
        "{help   |                  | print this message}"
        "{@image | ./book_cover.jpg | image path}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Load image file
    string filepath = parser.get<std::string>("@image");
    cv::Mat img = cv::imread(filepath);
    cv::Mat imgout;

    // Check that the image file was actually loaded
    if (img.empty()) {
        cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }

    cv::Vec3b black = cv::Vec3b(0,0,0);
    cv::Vec3b color = black;

    //drawRegtangle(img, color);


    //grayscale(img);
    cvtColor(img, imgout, cv::COLOR_RGB2GRAY);
    rectang(imgout, color);
    cv::imshow("Image", imgout);
    // Show the image



    // Wait for escape key press before returning
    while (cv::waitKey() != 27)
        ; // (do nothing)

    return 0;
}

void drawRegtangle(cv::Mat img, cv::Vec3b color){

  for (int i = 100; i < 350; i++) {
    img.at<cv::Vec3b>(220,i) = color;
  }

  for (int i = 100; i < 350; i++) {
    img.at<cv::Vec3b>(440,i) = color;
  }

  for (int i = 220; i < 440; i++) {
    img.at<cv::Vec3b>(i,100) = color;
  }

  for (int i = 220; i < 440; i++) {
    img.at<cv::Vec3b>(i,350) = color;
  }
}

void rectang(cv::Mat img, cv::Vec3b color){
  int x = 100;
  int y = 220;
  int width = 250;
  int height = 200 ;
  cv::Point pt1(x, y);
  cv::Point pt2(x + width, y + height);

  cv::rectangle(img, pt1, pt2, color);
}

void grayscale(cv::Mat img){
  for (int i = 0; i < img.size[0]; i++) {
    for (int j = 0; j < img.size[1]; j++) {
      //cout << "R " << (double)img.at<cv::Vec3b>(i,j).val[0] << ", G " << (double)img.at<cv::Vec3b>(i,j).val[1] << ", B " << (double)img.at<cv::Vec3b>(i,j).val[2]  << endl;
      img.at<cv::Vec3b>(i,j).val[0] = img.at<cv::Vec3b>(i,j).val[0] * 0.587; // B
      img.at<cv::Vec3b>(i,j).val[1] = img.at<cv::Vec3b>(i,j).val[1] * 0.114; // G
      img.at<cv::Vec3b>(i,j).val[2] = img.at<cv::Vec3b>(i,j).val[2] * 0.299; // R
      //cout << "R " << (double)img.at<cv::Vec3b>(i,j).val[0] << ", G " << (double)img.at<cv::Vec3b>(i,j).val[1] << ", B " << (double)img.at<cv::Vec3b>(i,j).val[2]  << endl;

    }
  }


}
