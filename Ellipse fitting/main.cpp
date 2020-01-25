#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void fitinEllipse(char* filename, int threshold);

int main(int argc, char** argv) {
	char* filename;
	filename = NULL;
	if (argc == 2) {
		filename = argv[1];
	}
	else {
		sprintf(filename, "test1.png");
	}


	fitinEllipse(filename, 150);
	cvWaitKey(0);
	destroyAllWindows();

	return 0;
}


void fitinEllipse(char* filename, int thresh) {

	Mat gray_img = imread(filename, IMREAD_GRAYSCALE);//读取灰度图
	Mat img = imread(filename);
	Mat result = imread(filename);//读取原图
	vector<vector<Point>> contours;
	Mat binary_img = gray_img >= thresh;//将读入的灰度图进行二值化
	findContours(binary_img, contours, RETR_LIST, CHAIN_APPROX_NONE);//检测二值化之后的图像轮廓（RETR_LIST参数，检查所有轮廓）
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> contour=contours[i];
		if (contour.size() < 6) continue;//如果少于6个点，直接放弃这个轮廓点
		RotatedRect box = cvFitEllipse2(&contour);//使用xvFitEllipse2进行椭圆拟合
		ellipse(result, box, Scalar(0, 255, 255), 1, CV_AA);//将拟合得到的椭圆绘制在原图上
	}//用椭圆拟合所有检测出的轮廓点
	
	imwrite("result.png", result);
	imshow(filename,img);
	imshow("result", result);
}
