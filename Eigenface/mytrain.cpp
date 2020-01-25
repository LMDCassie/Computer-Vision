#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;
IplImage* cutRoi(int id, int rows, int cols)
{
	char imagefile[25];
	char eyefile[25];
	sprintf(imagefile, "./train/BioID_%04d.pgm", id);
	sprintf(eyefile, "./train/BioID_%04d.eye", id);
	cout << imagefile << endl;
	cout << eyefile << endl;
	IplImage* image = cvLoadImage(imagefile);
	ifstream inf(eyefile);

	char buf[15];
	inf.getline(buf, 15);
	int lx, ly, rx, ry;
	// 读入眼睛坐标
	inf >> lx;
	inf >> ly;
	inf >> rx;
	inf >> ry;

	int x = rx - (lx - rx) / 2;
	int y = ry - (lx - rx);
	int w = 3.9 * (lx - rx) / 2;
	int h = 2.8 * (lx - rx);
	cout << x << " " << y << " " << w << " " << h << endl;
	inf.close();
	// 裁剪
	cvSetImageROI(image, cvRect(x, y, w, h));
	IplImage* dst = cvCreateImage(cvSize(w, h), image->depth, image->nChannels);
	cvCopy(image, dst);

	// 统一大小缩放
	IplImage* result = cvCreateImage(cvSize(cols, rows), image->depth, image->nChannels);
	cvResize(image, result);

	// 转化成单通道灰度图像
	IplImage* image_gray = cvCreateImage(cvGetSize(result), IPL_DEPTH_8U, 1);
	cvCvtColor(result, image_gray, CV_BGR2GRAY);

	cvEqualizeHist(image_gray, image_gray);
	if (id == 1) {
		cvShowImage("myself", image_gray);
	}
	return image_gray;

}

int main(int argc, char* argv[])
{
	int M = 40;
	int cols, rows;
	int N;
	float power = atof(argv[1]);  //EnergyPercent
	char* file_eigenface = argv[2];
	char* file_traincoeff = argv[3];
	char* file_meanface = argv[4];
	rows = 112;
	cols = 85;
	N = cols * rows;
	CvMat* S = cvCreateMat(N, M, CV_32FC1);

	if (argc != 5 || power <= 0 || power > 1) {
		cout << "Check the input." << endl;
		return -1;
	}

	for (int i = 1; i <= M; i++)
	{
		IplImage* image_gray = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1); //单通道灰度
		image_gray = cutRoi(i - 1, rows, cols);
		CvMat* img_mat = cvCreateMat(rows, cols, CV_16UC1);  //载入图片的矩阵形式
		cvConvert(image_gray, img_mat);  //把图片转化为矩阵		
		//reshape拉伸
		CvMat* oneCol, mathdr;
		oneCol = cvReshape(img_mat, &mathdr, 1, N); //矩阵reshape
		for (int j = 0; j < N; j++) {
			int value = cvGetReal2D(oneCol, j, 0);
			cvSetReal2D(S, j, i - 1, value);  
		}
	}

	//对S中列向量求和，进而求出平均人脸
	CvMat* mean_face = cvCreateMat(N, 1, CV_16UC1);
	cvSetZero(mean_face);
	for (int i = 0; i < M; i++) {
		CvMat* temp = cvCreateMat(N, 1, CV_16UC1); 
		cvGetCol(S, temp, i);
		cvAdd(temp, mean_face, mean_face);
	}

	cvConvertScale(mean_face, mean_face, 1.0 / M);

	//平均脸
	IplImage* mean_show = cvCreateImage(cvSize(rows, cols), IPL_DEPTH_8U, 1);
	IplImage mean_show_hdr;
	CvMat* mean_reshape = cvCreateMat(rows, cols, CV_8UC1);
	CvMat mean_hdr;
	mean_reshape = cvReshape(mean_face, &mean_hdr, 1, rows);  
	mean_show = cvGetImage(&mean_hdr, &mean_show_hdr);
	cvShowImage("mean", mean_show);
	cvSaveImage("./output/mean.jpg", mean_show);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			int S_value = cvGetReal2D(S, j, i);
			int mean_value = cvGetReal2D(mean_face, j, 0);
			cvSetReal2D(S, j, i, S_value - mean_value);
		}
	}
	// 求S的转置矩阵
	CvMat* ST = cvCreateMat(M, N, CV_32FC1);
	cvTranspose(S, ST);
	//求协方差矩阵S'S，大小为MM
	CvMat* Col_mat = cvCreateMat(M, M, CV_32FC1);
	cvMatMul(ST, S, Col_mat);  //矩阵相乘
	//构造输出特征向量矩阵
	CvMat* ProVector = cvCreateMat(M, M, CV_32FC1);
	//构造输出特征值矩阵
	CvMat* ProValue = cvCreateMat(M, 1, CV_32FC1);
	cvEigenVV(Col_mat, ProVector, ProValue, 1.0e-6F);
	//特征人脸空间
	CvMat* eigenface = cvCreateMat(N, int(M*power), CV_32FC1);
	//把ProVector的前 M*power列取出来
	CvMat* ProVector2 = cvCreateMat(M, int(M*power), CV_32FC1);
	for (int i = 0; i < int(M*power); i++) {
		for (int j = 0; j < M; j++) {
			float value = cvGetReal2D(ProVector, j, i);
			cvSetReal2D(ProVector2, j, i, value);
		}
	}
	//特征向量矩阵左乘S
	cvMatMul(S, ProVector2, eigenface);
	M = (int)M*power;

	cout << "----------" << endl;
	//对eigenface矩阵每一列归一化
	CvMat* tempVector = cvCreateMat(N, M, CV_32FC1);
	cvPow(eigenface, tempVector, 2);  
	CvMat* temp2 = cvCreateMat(1, M, CV_32FC1);
	cvReduce(tempVector, temp2, 0, CV_REDUCE_SUM);  
	for (int i = 0; i < M; i++) {   
		double sum;
		sum = cvGetReal2D(temp2, 0, i); 
		for (int j = 0; j < N; j++) { 
			double value;
			value = cvGetReal2D(eigenface, j, i);
			value = value / (sqrt(sum));
			cvSetReal2D(eigenface, j, i, value);

		}
	}

	//显示特征人脸
	CvMat* FirstCol = cvCreateMat(N, 1, CV_32FC1);
	IplImage* Add10 = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1);
	for (int k = 0; k < 10; k++) {
		for (int i = 0; i < N; i++) {
			float value = cvGetReal2D(eigenface, i, k);
			cvSetReal2D(FirstCol, i, 0, value);
		}
		cvNormalize(FirstCol, FirstCol, 255, 0, CV_MINMAX);
		CvMat* First_reshape = cvCreateMat(rows, cols, CV_32FC1);
		CvMat first_hdr;
		First_reshape = cvReshape(FirstCol, &first_hdr, 1, rows);
		IplImage* first_show = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_8U, 1);
		IplImage first_show_hdr;
		first_show = cvGetImage(&first_hdr, &first_show_hdr);
		char path[30];
		sprintf(path, "./output/col%d.jpg", k + 1);
		cvSaveImage(path, first_show);
	}
	cvNormalize(Add10, Add10, 255, 0, CV_MINMAX);
	cvShowImage("Add10", Add10);
	cvSaveImage("./output/Add10.jpg", Add10);

	CvMat* TrainCoeff = cvCreateMat(M, M, CV_32FC1);
	for (int i = 0; i < M; i++)
	{
		CvMat* temp = cvCreateMat(N, 1, CV_32FC1);
		cvGetCol(S, temp, i);
		CvMat* tempT = cvCreateMat(1, N, CV_32FC1);
		cvTranspose(temp, tempT);
		CvMat* c = cvCreateMat(1, M, CV_32FC1);
		cvMatMul(tempT, eigenface, c);
		CvMat* cT = cvCreateMat(M, 1, CV_32FC1);
		cvTranspose(c, cT);
		for (int j = 0; j < M; j++) {
			float value = cvGetReal2D(cT, j, 0);
			cvSetReal2D(TrainCoeff, j, i, value);
		}
	}

	//写入model文件
	cvSave(file_eigenface, eigenface);
	cvSave(file_traincoeff, TrainCoeff);
	cvSave(file_meanface, mean_face);

	cvWaitKey(0);
	cvReleaseImage(&Add10);

}
