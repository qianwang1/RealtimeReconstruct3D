#include "pylon/PylonIncludes.h"
#include <cv.hpp>
#include <highgui.h>


//定义是否保存图片
int saveImages = 0;

using namespace std;
using namespace Pylon;
using namespace cv;


//打开的相机数量
static const size_t c_maxCamerasToUse = 2;

cv::Mat readMatFromFile(string MatName);

void getRectifyImage(cv::Mat leftImage, cv::Mat& rectifyImageLeft, cv::Mat rightImage, cv::Mat& rectifyImageRight);

cv::Mat getDisparityImage(const cv::Mat rectifyImageLeft, const cv::Mat rectifyImageRight, cv::Mat &disp, int flag);


cv::Mat CameraLeftIntrix = readMatFromFile("CameraLeftIntrix");
cv::Mat CameraLeftDistCoeff = readMatFromFile("CameraLeftDistCoeff");

cv::Mat CameraRightIntrix = readMatFromFile("CameraRightIntrix");
cv::Mat CameraRightDistCoeff = readMatFromFile("CameraRightDistCoeff");

cv::Mat T = readMatFromFile("T");
cv::Mat R = readMatFromFile("R");

cv::Size imageSize = cv::Size(1280, 1024);
cv::Mat Rl, Rr, Pl, Pr, Q;
cv::Mat mapLeftX, mapLeftY, mapRightX, mapRightY;
cv::Rect validROILeft, validROIRight;

cv::Rect ROI(240, 212, 800, 600);
cv::Rect ROIDisp(260, 0, 540, 600);

cv::Mat rectifyImageLeft, rectifyImageRight;
cv::Mat disp;
cv::Mat disparity;
cv::Mat xyz;

void onMouseCallback(int event, int x, int y, int, void *);

cv::Point origin;
cv::Rect selection;
bool selectionObject;

int main()
{
    cv::stereoRectify(CameraLeftIntrix, CameraLeftDistCoeff, CameraRightIntrix, CameraRightDistCoeff, imageSize, R, T, Rl, Rr, Pl, Pr, Q, cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validROILeft, &validROIRight);
    initUndistortRectifyMap(CameraLeftIntrix, CameraLeftDistCoeff, Rl, Pl, imageSize, CV_32FC1, mapLeftX, mapLeftY);
    initUndistortRectifyMap(CameraRightIntrix, CameraRightDistCoeff, Rr, Pr, imageSize, CV_32FC1, mapRightX, mapRightY);
    cv::namedWindow("disparity");
    cv::setMouseCallback("disparity", onMouseCallback, 0);

    int exitCode = 0;

    //初始化Pylon
    PylonInitialize();

    try {
        //初始化Pylon的相机传输层
        CTlFactory& tlFactory = CTlFactory::GetInstance();

        //检测所有已附加的相机，如果没有找到则退出程序
        DeviceInfoList_t devices;
        if (tlFactory.EnumerateDevices(devices) == 0)
        {
            throw RUNTIME_EXCEPTION("No camera found!");
        }

        //创建相机实例数组，并防止超出最大相机数
        CInstantCameraArray cameras(min(devices.size(), c_maxCamerasToUse));
        cout << "The devices' size is: " << devices.size() << endl;

        //创建并附加相机
        for (size_t i = 0; i < cameras.GetSize(); ++i)
        {
            cameras[i].Attach(tlFactory.CreateDevice(devices[i]));

            //打印相机名称
            cout << "Using the device " << cameras[i].GetDeviceInfo().GetModelName() << endl;
        }


        //打开相机
        cameras[0].Open();
        cameras[1].Open();


        //设置相机缓冲，默认为10
        cameras[0].MaxNumBuffer = 5;
        cameras[1].MaxNumBuffer = 5;

        //创建Pylon的ImangeConvert对象，用来转换图片
        CImageFormatConverter formatConverter;
        //确定像素的输出格式
        formatConverter.OutputPixelFormat = PixelType_BGR8packed;
        //创建一个PylonImage，用来创建OpenCV Image
        CPylonImage pylonImageLeft;
        CPylonImage pylonImageRight;


        //新建OpenCV Image对象
        Mat ImageLeft;
        Mat ImageRight;


        //设置连续抓取模式
        cameras.StartGrabbing();

        //抓取结果的指针
        CGrabResultPtr ptrGrabResultLeft;
        CGrabResultPtr ptrGrabResultRight;

        //进入抓取图片阶段，抓取c_countOfImages个图片
        while (cameras[0].IsGrabbing() && cameras[1].IsGrabbing())
        {
            cameras[0].RetrieveResult(5000, ptrGrabResultLeft, TimeoutHandling_ThrowException);
            cameras[1].RetrieveResult(5000, ptrGrabResultRight, TimeoutHandling_ThrowException);

            if (ptrGrabResultLeft->GrabSucceeded() && ptrGrabResultRight->GrabSucceeded())
            {
                //将抓取的图像缓冲数据转换成PylonImage
                formatConverter.Convert(pylonImageLeft, ptrGrabResultLeft);
                formatConverter.Convert(pylonImageRight, ptrGrabResultRight);

                //然后将PylonIMage转换成OpenCVImage
                ImageLeft = Mat(ptrGrabResultLeft->GetHeight(), ptrGrabResultLeft->GetWidth(), CV_8UC3, (uint8_t *)pylonImageLeft.GetBuffer());
                ImageRight = Mat(ptrGrabResultRight->GetHeight(), ptrGrabResultRight->GetWidth(), CV_8UC3, (uint8_t *)pylonImageRight.GetBuffer());

                namedWindow("Left", 1);
                namedWindow("Right", 1);

                imshow("Left", ImageLeft);
                imshow("Right", ImageRight);


                getRectifyImage(ImageLeft, rectifyImageLeft, ImageRight, rectifyImageRight);
//                rectifyImageLeft = Mat(rectifyImageLeft, ROI);
//                rectifyImageRight = Mat(rectifyImageRight, ROI);

                disparity = getDisparityImage(rectifyImageLeft, rectifyImageRight, disp, 1);

                cv::reprojectImageTo3D(disp, xyz, Q, true);
                cv::imshow("disparity", disparity);

                char c = cv::waitKey(10);
                if (c == 'q')
                    break;
            }
        }
    }
    catch (const GenericException& e)
    {
        cerr << "An exception occur red." << endl << e.GetDescription() << endl;
        exitCode = 1;
    }

    //释放Pylon
    PylonTerminate();

    return exitCode;
}


cv::Mat readMatFromFile(string MatName)
{
    cv::Mat tempMat;
    string FileName = "../data/" + MatName + ".yml";
    FileStorage fileStorage(FileName, FileStorage::READ);
    fileStorage[MatName] >> tempMat;
    return tempMat;
}

void getRectifyImage(cv::Mat leftImage, cv::Mat& rectifyImageLeft, cv::Mat rightImage, cv::Mat& rectifyImageRight)
{
    cv::Mat grayImageLeft, grayImageRight;
    cv::cvtColor(leftImage, grayImageLeft, CV_RGB2GRAY);
    cv::cvtColor(rightImage, grayImageRight, CV_RGB2GRAY);
    cv::remap(grayImageLeft, rectifyImageLeft, mapLeftX, mapLeftY, cv::INTER_LINEAR);
    cv::remap(grayImageRight, rectifyImageRight, mapRightX, mapRightY, cv::INTER_LINEAR);
    rectifyImageLeft = Mat(rectifyImageLeft, validROILeft);
    rectifyImageRight = Mat(rectifyImageRight, validROIRight);
//    rectifyImageLeft = Mat(rectifyImageLeft, ROI);
//    rectifyImageRight = Mat(rectifyImageRight, ROI);
}

cv::Mat getDisparityImage(const cv::Mat rectifyImageLeft, const cv::Mat rectifyImageRight, cv::Mat &disp, int flag) {
    if (flag == 1) {
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
        sgbm->setPreFilterCap(63);
        sgbm->setBlockSize(5);
        int cn = rectifyImageLeft.channels();

        sgbm->setP1(8 * cn * 5 * 5);
        sgbm->setP2(32 * cn * 5 * 5);
        sgbm->setMinDisparity(0);
        sgbm->setNumDisparities(256);
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(100);
        sgbm->setSpeckleRange(32);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

        Mat disp8;

        sgbm->compute(rectifyImageLeft, rectifyImageRight, disp);

        disp.convertTo(disp8, CV_8U, 255 / (256 * 16.));

        return disp8;
    } else if (flag == 2) {
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
        bm->setROI1(validROILeft);
        bm->setROI2(validROIRight);
        bm->setPreFilterCap(31);
        bm->setBlockSize(5);
        bm->setMinDisparity(0);
        bm->setNumDisparities(256);
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(15);
        bm->setSpeckleRange(100);
        bm->setDisp12MaxDiff(1);
        Mat disp, disp8;
        bm->compute(rectifyImageLeft, rectifyImageRight, disp);

        disp.convertTo(disp8, CV_8U, 255 / (256 * 16.));
        return disp8;
    }
}


void onMouseCallback(int event, int x, int y, int, void *) {
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            origin = cv::Point(x, y);
            selection = cv::Rect(x, y, 0, 0);
            selectionObject = true;
            float worldx = xyz.at<Vec3f>(x, y)[0];
            float worldy = xyz.at<Vec3f>(x, y)[1];
            float worldz = xyz.at<Vec3f>(x, y)[2];
            float distance = sqrt(worldx * worldx + worldy * worldy + worldz * worldz);
            std::cout << worldx << " " << worldy << " " << worldz << std::endl;
            std::cout << distance << std::endl;
            break;

    }
}