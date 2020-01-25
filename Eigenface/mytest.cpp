#include <cv.h>
#include <highgui.h>
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 5) {
		cout << "wrong input" << endl;
		return -1;
	}

	char* filename = argv[1];
	CvMat* eigenface = (CvMat*)cvLoad(argv[2]);
	CvMat* traincoeff = (CvMat*)cvLoad(argv[3]);
	CvMat* mean = (CvMat*)cvLoad(argv[4]);  
	int N = cvGetSize(eigenface).height;
	int M = cvGetSize(eigenface).width;

	IplImage* input = cvLoadImage(filename);
	IplImage* input_gray = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvCvtColor(input, input_gray, CV_BGR2GRAY);

	cvEqualizeHist(input_gray, input_gray); 

	int rows = cvGetSize(input).height;
	int cols = cvGetSize(input).width;
	CvMat* input_mat = cvCreateMat(rows, cols, CV_32FC1);
	cvConvert(input_gray, input_mat);

	// 把input_mat拉伸成列向量
	CvMat* input_n, mathdr;
	input_n = cvReshape(input_mat, &mathdr, 1, N);
	// 拉伸后的列向量减去平均人脸,使用元素级相减函数
	cvSub(input_n, mean, input_n);

	//计算输入照片在基底上的投影坐标
	CvMat* input_n_T = cvCreateMat(1, N, CV_32FC1);
	cvTranspose(input_n, input_n_T);

	CvMat* coeff = cvCreateMat(1, M, CV_32FC1); //投影后坐标
	cvMatMul(input_n_T, eigenface, coeff);

	CvMat* dist = cvCreateMat(1, M, CV_32FC1); //输入图像与每张训练图像的欧式距离
	CvMat* coeffT = cvCreateMat(M, 1, CV_32FC1);
	cvTranspose(coeff, coeffT);
	for (int i = 0; i < M; i++) {
		//计算输入图片投影后的坐标与traincoeff中的每列的欧式举例
		CvMat* train_each = cvCreateMat(M, 1, CV_32FC1);  

		for (int j = 0; j < M; j++) {
			float value = cvGetReal2D(traincoeff, j, i);
			cvSetReal2D(train_each, j, 0, value);
		}
		CvMat* coeffTtemp = cvCreateMat(M, 1, CV_32FC1);
		cvSub(coeffT, train_each, coeffTtemp);
		float distance = 0;
		for (int j = 0; j < M; j++) {
			float value = cvGetReal2D(coeffTtemp, j, 0);
			distance += value * value;
		}
		cvSetReal2D(dist, 0, i, sqrt(distance));
	}

	// 找到距离最小值
	CvPoint min_loc = cvPoint(0, 0);
	CvPoint max_loc = cvPoint(0, 0);
	double min_value, max_value;
	cvMinMaxLoc(dist, &min_value, &max_value, &min_loc, &max_loc);

	int id = (&min_loc)->x;  // 最小距离位于第几列，是人脸库中最相近的人脸
	cout << "和第" << id << "张最像" << endl;

	CvMat* reconstruct = cvCreateMat(N, 1, CV_32FC1);
	cvMatMul(eigenface, coeffT, reconstruct);
	cvAdd(reconstruct, mean, reconstruct); 

	// 把reconstruct映射到0-255上
	cvNormalize(reconstruct, reconstruct, 255, 0, CV_MINMAX);
	CvMat* reconstruct_reshape = cvCreateMat(rows, cols, CV_32FC1);
	CvMat reconstruct_hdr;
	reconstruct_reshape = cvReshape(reconstruct, &reconstruct_hdr, 1, rows);
	IplImage* reconstruct_show = cvCreateImage(cvSize(rows, cols), IPL_DEPTH_8U, 1);
	IplImage reconstruct_show_hdr;
	reconstruct_show = cvGetImage(&reconstruct_hdr, &reconstruct_show_hdr);
	//cvShowImage("reconstruct", reconstruct_show);
	cvSaveImage("./output/reconstruct.jpg", reconstruct_show);
	cvShowImage("input", input);

	// 重构结果叠加到输入的人脸上
	IplImage* reconstruct_add_input = cvCreateImage(cvGetSize(reconstruct_show), IPL_DEPTH_8U, 1);
	cvAddWeighted(input_gray, 0.5, reconstruct_show, 0.5, 0, reconstruct_add_input);
	cvShowImage("addweight", reconstruct_add_input);

	//显示最像的图片
	IplImage* most_likely;
	char likely_filename[50];
	sprintf(likely_filename, "./train2/BioID_%04d.pgm", id);
	most_likely = cvLoadImage(likely_filename);
	cvShowImage("most likely image", most_likely);

	cvWaitKey(0);
	cvReleaseImage(&reconstruct_show);
	cvReleaseImage(&input);
	cvReleaseImage(&input_gray);
	cvReleaseImage(&reconstruct_show);
	cvReleaseImage(&reconstruct_add_input);

}
