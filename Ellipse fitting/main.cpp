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

	Mat gray_img = imread(filename, IMREAD_GRAYSCALE);//��ȡ�Ҷ�ͼ
	Mat img = imread(filename);
	Mat result = imread(filename);//��ȡԭͼ
	vector<vector<Point>> contours;
	Mat binary_img = gray_img >= thresh;//������ĻҶ�ͼ���ж�ֵ��
	findContours(binary_img, contours, RETR_LIST, CHAIN_APPROX_NONE);//����ֵ��֮���ͼ��������RETR_LIST�������������������
	for (int i = 0; i < contours.size(); i++) {
		vector<Point> contour=contours[i];
		if (contour.size() < 6) continue;//�������6���㣬ֱ�ӷ������������
		RotatedRect box = cvFitEllipse2(&contour);//ʹ��xvFitEllipse2������Բ���
		ellipse(result, box, Scalar(0, 255, 255), 1, CV_AA);//����ϵõ�����Բ������ԭͼ��
	}//����Բ������м�����������
	
	imwrite("result.png", result);
	imshow(filename,img);
	imshow("result", result);
}
