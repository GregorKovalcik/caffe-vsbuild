#include "CaffeFeatureExtractorCLR.hpp"

namespace CaffeFeatureExtractorCLR
{
    CaffeFeatureExtractor::CaffeFeatureExtractor(String^ modelFile, String^ trainedFile, String^ meanFile)
    {
        mFeatureExtractor = new FeatureExtractor(
            marshal_as<std::string>(modelFile),
            marshal_as<std::string>(trainedFile),
            marshal_as<std::string>(meanFile));

        SetModeGPU();
    }

    CaffeFeatureExtractor::~CaffeFeatureExtractor()
    {
        delete mFeatureExtractor;
        mFeatureExtractor = NULL;
    }

    void CaffeFeatureExtractor::SetModeGPU()
    {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
    }

    void CaffeFeatureExtractor::SetModeCPU()
    {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }

    int CaffeFeatureExtractor::GetInputWidth()
    {
        return mFeatureExtractor->GetInputSize().width;
    }

    int CaffeFeatureExtractor::GetInputHeight()
    {
        return mFeatureExtractor->GetInputSize().height;
    }

    void CaffeFeatureExtractor::ForwardImage(Bitmap^ image)
    {
        Bitmap^ resized = image;
        if (image == nullptr)
        {
            throw gcnew System::Exception(gcnew System::String("Bitmap is a null pointer!"));
        }
        else if (image->Width != GetInputWidth() || image->Height != GetInputHeight())
        {
            resized = gcnew Bitmap(image, GetInputWidth(), GetInputHeight());
        }
        mFeatureExtractor->ForwardImage(ConvertBitmapToMat(resized));
    }

    array<float>^ CaffeFeatureExtractor::ExtractFeatures(String^ blobName)
    {
        cv::Mat feature = mFeatureExtractor->ExtractFeatures(marshal_as<std::string>(blobName));
        array<float>^ result = gcnew array<float>(feature.cols);
        System::Runtime::InteropServices::Marshal::Copy(IntPtr((void *)feature.data), result, 0, feature.cols);
        return result;
    }

    array<float>^ CaffeFeatureExtractor::ExtractFromImage(Bitmap^ image, String^ blobName)
    {
        ForwardImage(image);
        return ExtractFeatures(blobName);
    }

    Bitmap^ CaffeFeatureExtractor::ConvertMatToBitmap(cv::Mat mat)
    {
        switch (mat.channels())
        {
        case 1:
            return gcnew Bitmap(
                mat.cols,
                mat.rows,
                mat.step,
                System::Drawing::Imaging::PixelFormat::Format8bppIndexed,
                IntPtr(mat.data));
        case 3:
            return gcnew Bitmap(
                mat.cols,
                mat.rows,
                mat.step,
                System::Drawing::Imaging::PixelFormat::Format24bppRgb,
                IntPtr(mat.data));
        case 4:
            return gcnew Bitmap(
                mat.cols,
                mat.rows,
                mat.step,
                System::Drawing::Imaging::PixelFormat::Format32bppArgb,
                IntPtr(mat.data));
        default:
            throw gcnew System::Exception(gcnew System::String("Unsupported cv::Mat format!"));
        }
    }

    cv::Mat CaffeFeatureExtractor::ConvertBitmapToMat(Bitmap^ bitmap)
    {
        IplImage* iplImage;

        BitmapData^ bitmapData = bitmap->LockBits(
            System::Drawing::Rectangle(0, 0, bitmap->Width, bitmap->Height),
            ImageLockMode::ReadWrite,
            bitmap->PixelFormat);

        if (bitmap->PixelFormat == PixelFormat::Format8bppIndexed)
        {
            iplImage = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 1);
            iplImage->imageData = (char*)bitmapData->Scan0.ToPointer();
        }
        else if (bitmap->PixelFormat == PixelFormat::Format24bppRgb)
        {
            iplImage = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 3);
            iplImage->imageData = (char*)bitmapData->Scan0.ToPointer();
        }
        else if (bitmap->PixelFormat == PixelFormat::Format32bppRgb
            || bitmap->PixelFormat == PixelFormat::Format32bppArgb)
        {
            iplImage = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 4);
            iplImage->imageData = (char*)bitmapData->Scan0.ToPointer();
        }
        else
        {
            throw gcnew System::Exception(gcnew System::String("Unsupported bitmap format!"));
        }
        bitmap->UnlockBits(bitmapData);

        cv::Mat mat = cv::cvarrToMat(iplImage);
        cvReleaseImage(&iplImage);
        return mat;
    }

    Bitmap^ CaffeFeatureExtractor::TestBitmapConversion(Bitmap^ bitmap)
    {
        return ConvertMatToBitmap(ConvertBitmapToMat(bitmap));
    }
}
