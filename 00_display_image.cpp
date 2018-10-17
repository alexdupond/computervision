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

void show_image(Mat img, string name);
Mat histogram(const Mat& img);
Mat draw_histogram(const Mat& hist);
Mat I2H(Mat img){
    return draw_histogram(histogram(img));
}
void FreqFilt(Mat img, float d0, int n);
void dftshift(cv::Mat& mag);
void show_complex(Mat complex, string name);
Mat butter_lowpass(float d0, int n, cv::Size size);
Mat notch_filter(float area, int freksX , int freksY, Size size);
void FreqFiltNotch(Mat img, float area, int freksX,int freksY);
void pic3(Mat img);

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

    // Original img.
//    show_image(img0, "Original Don't use");

/***************************************************************************
*                                                                          *
*                           Image 4_1                                      *
*                                                                          *
***************************************************************************/
/*
    show_image(img4_1, "img4_1 org");
    show_image(I2H(img4_1), "histogram org");

    Mat img = img4_1;

    Rect ROI(700,1400,800,400);
    img4_1 = img4_1(ROI);
    show_image(img4_1,"ROI Org");
*/
    //FreqFilt(img, 250, 100); // Diameter, n
   // FreqFiltNotch(img,50,2140,1760);


//    Mat hist = histogram(img1);
    //  imshow("Histgram Orgi", draw_histogram(hist));
    // imshow("Histgram Filtered", draw_histogram(hist));

//    Mat eqimg;
//    equalizeHist(img1, eqimg);

// IMG 3
   show_image(img3, "img3");
   namedWindow("hist org", WINDOW_NORMAL);
   imshow("hist org",draw_histogram(histogram(img3)));

    pic3(img3);

/***************************************************************************
*                                                                          *
*                           Exit routeine                                  *
*                                                                          *
***************************************************************************/
  while (cv::waitKey() != 27) // Wait for esc key
      ; // (do nothing)
}
void FreqFiltNotch(Mat img, float area, int freksX,int freksY){
    cv::Mat padded;
        int opt_rows = cv::getOptimalDFTSize(img.rows * 2 - 1);
        int opt_cols = cv::getOptimalDFTSize(img.cols * 2 - 1);
        cv::copyMakeBorder(img, padded, 0, opt_rows - img.rows, 0, opt_cols - img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[] = {
            cv::Mat_<float>(padded),
            cv::Mat_<float>::zeros(padded.size())
        };
        cv::Mat complex;
        cv::merge(planes, 2, complex);

        // Compute DFT of image
        cv::dft(complex, complex);
      show_complex(complex, "origin");

        // Shift quadrants to center
        dftshift(complex);

        //Øverst højre hjørne
        Mat filter = notch_filter( area, freksX, freksY , complex.size());
        // Multiply Fourier image with filter
        cv::mulSpectrums(complex, filter, complex, 0);

        //Øverste venstre hjørne
        filter = notch_filter( area, 1350, 2200, complex.size());
        cout << "freksX X: " << freksX+1324 << " | Y: " << freksY-1175 << endl;
        // Multiply Fourier image with filter
        cv::mulSpectrums(complex, filter, complex, 0);

        //Neders højre hjørne
        filter = notch_filter( area, 1740 , 2640, complex.size());
        cout << "freksX X: " << freksY-350 << " | Y: " << freksX+924 << endl;
        // Multiply Fourier image with filter
        cv::mulSpectrums(complex, filter, complex, 0);

        filter = notch_filter( area, 925, 3064, complex.size());
        cout << "freksX X: " << freksY-700 << " | Y: " << freksX+400 << endl;
        // Multiply Fourier image with filter
        cv::mulSpectrums(complex, filter, complex, 0);

        // Shift back
        dftshift(complex);
        show_complex(complex, "filtered");
        // Compute inverse DFT
        cv::Mat filtered;
        cv::idft(complex, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

        // Crop image (remove padded borders)
        filtered = cv::Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));

        dftshift(filter);
//        show_complex(filter, "Filter");

        cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
        show_image(filtered, "filtered image");

        Rect ROI(700,1400,800,400);
        filtered = filtered(ROI);
        show_image(filtered,"ROI");

        //I2H(filtered);
}

//KIM
void FreqFilt(Mat img, float d0, int n){
    cv::Mat padded;
        int opt_rows = cv::getOptimalDFTSize(img.rows * 2 - 1);
        int opt_cols = cv::getOptimalDFTSize(img.cols * 2 - 1);
        cv::copyMakeBorder(img, padded, 0, opt_rows - img.rows, 0, opt_cols - img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[] = {
            cv::Mat_<float>(padded),
            cv::Mat_<float>::zeros(padded.size())
        };
        cv::Mat complex;
        cv::merge(planes, 2, complex);

        // Compute DFT of image
        cv::dft(complex, complex);
      show_complex(complex, "origin");

        // Shift quadrants to center
        dftshift(complex);

        Mat filter = butter_lowpass(d0, n, complex.size());

        // Multiply Fourier image with filter
        cv::mulSpectrums(complex, filter, complex, 0);

        // Shift back
        dftshift(complex);
        show_complex(complex, "filtered");
        // Compute inverse DFT
        cv::Mat filtered;
        cv::idft(complex, filtered, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

        // Crop image (remove padded borders)
        filtered = cv::Mat(filtered, cv::Rect(cv::Point(0, 0), img.size()));

        dftshift(filter);
//        show_complex(filter, "Filter");

        cv::normalize(filtered, filtered, 0, 1, cv::NORM_MINMAX);
        show_image(filtered, "filtered image");
        //I2H(filtered);
}
// KIM
Mat butter_lowpass(float d0, int n, Size size)
{
    Mat_<Vec2f> lpf(size);
    Point2f c = cv::Point2f(size) / 2;

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            // Distance from point (i,j) to the origin of the Fourier transform
            float d = sqrt((i - c.y) * (i - c.y) + (j - c.x) * (j - c.x));

            // Real part
            lpf(i, j)[0] = 1 / (1 + std::pow(d / d0, 2 * n));

            // Imaginary part
            lpf(i, j)[1] = 0;
        }
    }

    return lpf;
}
cv::Mat notch_filter(float area, int freksX , int freksY, Size size){
    Mat_<Vec2f> nFilter(size);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
             nFilter(i, j)[0] =1;
            if( i>freksY-area && i<freksY+area)
            {
                if (j>freksX-area && j<freksX+area)
                {
                    nFilter(i, j)[0] =0;
                   // cout << "i,j: " << i << ", " << j <<  endl;
                }
            }
            nFilter(i, j)[1] = 0;
        }
    }
    return nFilter;
}


// https://vgg.fiit.stuba.sk/2012-05/frequency-domain-filtration/
// KIM // Rearranges the quadrants of a Fourier image so that the origin is at the
// center of the image.
void dftshift(cv::Mat& mag)
{
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    cv::Mat tmp;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void show_complex(Mat complex, string name){
    Mat magI;
    Mat planes[] = {
        Mat::zeros(complex.size(), CV_32F),
        Mat::zeros(complex.size(), CV_32F)
    };
    split(complex, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], magI); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    // switch to logarithmic scale: log(1 + magnitude)
    magI += Scalar::all(1);
    log(magI, magI);
    dftshift(magI); // rearrage quadrants
    // Transform the magnitude matrix into a viewable image (float values 0-1)
    normalize(magI, magI, 1, 0, NORM_INF);
    namedWindow(name, WINDOW_NORMAL);
    imshow(name, magI);
}


void show_image(Mat img, string name){
  namedWindow(name, WINDOW_NORMAL);
  imshow(name, img);
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

void pic3(Mat img)
{
    // Filter inspired by https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html
    int kernel_size=30;
    Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth =-1;
    Mat img1Filt;
    Mat img1Edit;
    img1Edit=img;


    Mat kernel = cv::Mat::ones(kernel_size,kernel_size, CV_32F)/ (float)(kernel_size*kernel_size);
  //  cv::filter2D(img1Edit, img1Filt, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
    cv::bilateralFilter(img1Edit, img1Filt, 5, 50, 50);

//    namedWindow("Linear filter", WINDOW_NORMAL);
//    imshow("Linear filter",img1Filt);

    Mat frame;
    cv::GaussianBlur( img1Filt,frame, cvSize(0, 0), 3);

//    namedWindow("frame", WINDOW_NORMAL);
//    imshow("frame",frame);

    cv::addWeighted(img1Filt, 1.4 ,frame , -0.5, 0, img1Filt);

    namedWindow("Filtered", WINDOW_NORMAL);
    imshow ("Filtered",img1Filt);

    namedWindow("hist filtered", WINDOW_NORMAL);
    imshow("hist filtered",draw_histogram(histogram(img1Filt)));
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
