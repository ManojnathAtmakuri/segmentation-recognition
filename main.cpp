#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include<math.h>

using namespace cv;
using namespace std;

Mat image;
Mat EM;
Mat label_img;
Mat canny_img;
Mat Hough_img;
Mat Emeans;
int x0, x1;
vector<Vec3f> Circle;

void K_means(int a, int b)
{
	int x0 = 0;
	int x1 = 0;
	bool t = false;
	while (!t){
		int s0 = 0;
		int s1 = 0;
		int v0 = 0;
		int v1 = 0;
		
		for (int i = 0; i < image.cols; i++){
			for (int j = 0; j < image.rows; j++){
				int f0 = abs(image.ptr<uchar>(j)[i] - a);
				int f1 = abs(image.ptr<uchar>(j)[i] - b);
				if (f0 < f1){
					label_img.ptr<uchar>(j)[i] = 0;
					s0 += image.ptr<uchar>(j)[i];
					v0++;
				}
				else{
					label_img.ptr <uchar>(j)[i] = 255;
					s1 += image.ptr<uchar>(j)[i];
					v1++;
				}
			}
		}
		if (v0 != 0)
		{
			x0 = s0 / v0;
		}
		if (v1 != 0)
		{
			x1 = s1 / v1;
		}
		if (abs(a - x0) < 0.50 && abs(b - x1)< 0.50)
		{
			t = true;
		}
		a = x0;
		b = x1;
	}
}

void expectation_maximization(const Mat img,const Mat label, Mat* l_img){

	
	float me0;
	float me1;
	float l_me0;
	float l_me1;
	float add_1;
	float add_2;
	float k0;
	float k1;
	float z0;
	float z1;
	float n_me0;
	float n_me1;
	float n_z0;
	float n_z1;
	
	for (int i = 0; i < image.cols; i++){
		for (int j = 0; j < image.rows; j++){
			if (label_img.ptr<uchar>(j)[i] > 0)
			{
				EM.ptr<float>(j)[i] = 1;
			}
			else{
				EM.ptr<float>(j)[i] = 0;
			}
		}
	}

	int end = 1;
	me0 = x0;
	me1 = x1;
	while (end){

		l_me0 = 0;
		l_me1 = 0;
		add_1 = 0;
		add_2 = 0;

		for (int i = 0; i < img.cols; i++){
			for (int j = 0; j < img.rows; j++){

				l_me0 += pow(image.ptr<uchar>(j)[i] - me0, 2) * EM.ptr<float>(j)[i];
				add_1 += EM.ptr<float>(j)[i];

				l_me1 += pow(image.ptr<uchar>(j)[i] - me1, 2) * abs(1 - EM.ptr<float>(j)[i]);
				add_2 += abs(1 - EM.ptr<float>(j)[i]);
			}
		}

		z0 = sqrt((float)(l_me0 / add_1));
		z1 = sqrt((float)(l_me1 / add_2));

		n_me0 = 0;
		n_z0 = 0;
		n_me1 = 0;
		n_z1 = 0;

		for (int i = 0; i < image.cols; i++){
			for (int j = 0; j < image.rows; j++){

				k0 = (1 / sqrt(2 * 3.14 * pow(z0, 2))) * exp(-pow(image.ptr<uchar>(j)[i] - me0, 2) / (2 * pow(z0, 2)));
				k1 = (1 / sqrt(2 * 3.14 * pow(z1, 2))) * exp(-pow(image.ptr<uchar>(j)[i] - me1, 2) / (2 * pow(z1, 2)));
				EM.ptr<float>(j)[i] = k0 / (k0 + k1);;
				n_me0 += EM.ptr<float>(j)[i] * image.ptr<uchar>(j)[i];
				n_z0 += EM.ptr<float>(j)[i];
				n_me1 += abs(1 - EM.ptr<float>(j)[i]) * image.ptr<uchar>(j)[i];
				n_z1 += abs(1 - EM.ptr<float>(j)[i]);
			}
		}

		if (abs(me0 - (n_me0 / n_z0)) < 0.30){
			end = 0;
		}
		me0 = n_me0 / n_z0;
		me1 = n_me1 / n_z1;
	}

	for (int i = 0; i < img.cols; i++){
		for (int j = 0; j < img.rows; j++){
			if (EM.ptr<float>(j)[i] >= 0.5) {
				l_img->ptr<uchar>(j)[i] = 255;
			}
			else{
				l_img->ptr<uchar>(j)[i] = 0;
			}
		}
	}
}
void hough()
{
	Circle.clear();
	HoughCircles(canny_img, Circle, CV_HOUGH_GRADIENT, 1, canny_img.rows / 4, 200, 50, 100, 0);
	Mat channel[3];
	for (int i = 0; i < 3; i++)
	{
		canny_img.copyTo(channel[i]);
	}
	merge(channel, 3, Hough_img);
	for (size_t i = 0; i < Circle.size(); i++)
	{
		Point cnter(cvRound(Circle[i][0]), cvRound(Circle[i][1]));
		int radius = cvRound(Circle[i][2]);
		circle(Hough_img, cnter, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(Hough_img, cnter, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
}

int main(int argc, char ** argv){

	image = imread("Original+Image.png", 0);
	resize(image, image, Size(image.cols / 2, image.rows / 2));
	label_img.create(image.rows, image.cols, CV_8UC1);  
	EM.create(image.rows, image.cols, CV_32FC1);
	Emeans.create(image.rows, image.cols, CV_8UC1);
	Hough_img.create(image.rows, image.cols, CV_8UC1);
	K_means(x0, x1);
	expectation_maximization(image,label_img, &Emeans);
	Canny(label_img, canny_img, 1, 2, 3);
	hough();
	while (1){
		imshow("Orginal_Image", image);
		imshow("Canny", canny_img);
		imshow("Hough", Hough_img);
		imshow("K-means", label_img);
		imshow("EM", Emeans);
	
		char c = waitKey(10);
		if (c == 27)
		{
			break;
		}
	}

	return 0;

}

