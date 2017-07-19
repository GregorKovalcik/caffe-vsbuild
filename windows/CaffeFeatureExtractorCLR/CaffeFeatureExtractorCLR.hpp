#pragma once

#include "../caffe_feature_extractor_lib/caffe_feature_extractor_lib.hpp"
#include <msclr/marshal_cppstd.h>
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>

using namespace System;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
using namespace msclr::interop;

namespace CaffeFeatureExtractorCLR
{
    /// <summary>
    /// C++/CLI wrapper for Caffe feature extractor
    /// </summary>
    public ref class CaffeFeatureExtractor
    {
    private:
        FeatureExtractor* mFeatureExtractor;

    public:
        /// <summary>
        /// modelFile == deploy.prototxt
        /// trainedFile == [name].caffemodel
        /// meanFile == mean.binaryproto
        /// </summary>
        CaffeFeatureExtractor(String^ modelFile, String^ trainedFile, String^ meanFile);
        ~CaffeFeatureExtractor();

        void SetModeGPU();
        void SetModeCPU();

        int GetInputWidth();
        int GetInputHeight();

        void ForwardImage(Bitmap^ image);
        array<float>^ ExtractFeatures(String^ blobName);
        array<float>^ ExtractFromImage(Bitmap^ image, String^ blobName);

        static Bitmap^ TestBitmapConversion(Bitmap^ bitmap);

    private:
        static Bitmap^ ConvertMatToBitmap(cv::Mat mat);

        static cv::Mat ConvertBitmapToMat(Bitmap^ bitmap);

    };
}
