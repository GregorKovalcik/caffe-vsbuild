#ifndef CAFFE_FEATURE_EXTRACTOR_LIB_OUTPUT_MODULE_HPP
#define CAFFE_FEATURE_EXTRACTOR_LIB_OUTPUT_MODULE_HPP

#define XML_MAT_IDENTIFIER "caffe_features"

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>

/**
* @brief Output module implementing text, XML and multiple image outputs.
*
* Text output:
*   Writes features to a text file in following format:
*   <path_to_input_file>:<float1>;<float2>;...;<floatN>
* XML output:
*   Buffers features in a cv::Mat, then serializes it using cv::FileStorage
*   with identifier "caffe_features"
* Image output:
*   Generates four different image files:
*   - grayscale image normalized to 0..1 range
*   - high contrast version of above normalized image, where zero values are copied
*     and positive values are set to 1.
*   - RGB image normalized to -1..+1 range, where negative values are in blue shade
*     and positive values are in red shade
*   - high contrast version of above normalized image, where zero values are copied,
*     negative values are set to RGB(0,0,255) and positive values are set to RGB(255,0,0)
*   All image files are saved as a 8BPP PNG file using range 0..255.
*
*   It is possible to set maximal image height. Higher images are then split into smaller files
*   and saved and the memory is released (reused).
*/
class OutputModule
{
public:

    /**
    * @brief Create output module, initialize output paths for different output files,
     * and open output streams.
     */
    OutputModule(
        const boost::shared_ptr<caffe::Net<float>>& net,
        const boost::filesystem::path& outputPath,
        const std::string& blobName,
        bool enableTextOutput = true,
        bool enableImageOutput = false,
        bool enableXmlOutput = false,
        bool enableBinaryOutput = false,
        int numberOfImageRows = 0,
        bool enableKernelMaxPooling = false);

    /**
    * @brief Close output module on destruction if user forgot to do it himself.
    */
    ~OutputModule()
    {
        if (!mIsClosed)
        {
            Close();
        }
    }

    /**
    * @brief Write feature for given filename, using blob that was preloaded in constructor
    */
    void WriteFeatureFor(const std::string& inputFilename)
    {
        WriteText(inputFilename);
        WriteBinary(inputFilename);
        WriteImage(inputFilename);
        mWrittenFeatureCount++;
    }

    /**
    * @brief Write all unsaved data.
    */
    void Close();


    bool isTextOutputEnabled()
    {
        return mIsTextOutputEnabled;
    }

    bool isImageOutputEnabled()
    {
        return mIsImageOutputEnabled;
    }

    bool isXMLOutputEnabled()
    {
        return mIsXMLOutputEnabled;
    }

    bool isBinaryOutputEnabled()
    {
        return mIsBinaryOutputEnabled;
    }


private:
    bool mIsTextOutputEnabled;
    bool mIsImageOutputEnabled;
    bool mIsXMLOutputEnabled;
    bool mIsBinaryOutputEnabled;
    int mImageMaxHeight;	    // 0 -> do not split the image

    bool mIsKernelMaxPoolingEnabled;    // compute maximal activation over all results of one kernel
                                        // resulting feature dimension is equal to blob channel dimension
                                        // (number of layer kernels)

    boost::shared_ptr<caffe::Blob<float>> mBlob;

    std::string mOutputPath;
    std::string mOutputPathBinary;
    std::ofstream mOutputStream;
    std::ofstream mOutputStreamBinary;

    std::string mOutputPathStripped;	// without extension, used to create custom filename for different output files
    cv::Mat mOutputImage;                   // image normalized into full range [0..255]
    cv::Mat mOutputImageContrast;           // higher contrast image, where all nonzero values are 255 and zero values remain 0
    cv::Mat mOutputImageBlueRed;            // normalized, but negative values displayed in blue, while positive are displayed in red
    cv::Mat mOutputImageBlueRedContrast;    // similar high contrast in 3 colors -> positive values are bright red, negative are blue
                                            // and zeros are preserved
    cv::Mat mOutputXML;     // XML output, values are saved in float (CV_32F) as they were extracted.

    int mFileCounter = 0;   // used when splitting into multiple files
    int32_t mWrittenFeatureCount = 0;
    bool mIsClosed = false;

    /**
    * @brief Write text output into std::ofstream using a vector wrapped around the blob data.
    */
    void WriteText(const std::string& inputFilename);

    /**
    * @brief Write text output into std::ofstream using a vector wrapped around the blob data.
    */
    void WriteBinary(const std::string& inputFilename);

    /**
    * @brief Write image and XML output using a cv::Mat wrapped around the blob data.
    */
    void WriteImage(const std::string& inputFilename);

    /**
    * @brief Preprocess image and append to image buffers
    */
    void NormalizeAndSaveFeature(const cv::Mat& featureImage);

    /**
    * @brief Replaces illegal characters from string so it can be used as filename.
    */
    static std::string ReplaceIllegalCharacters(const std::string& str)
    {
        std::string strSafe(str);
        std::string illegalChars = "\\/:?\"<>|";
        for (auto it = strSafe.begin(); it < strSafe.end(); ++it){
            bool found = illegalChars.find(*it) != std::string::npos;
            if (found)
            {
                *it = '_';
            }
        }
        return strSafe;
    }

    /**
    * @brief Saves the buffered image and resets its line count to 0,
    *       so new lines are written from start again.
    */
    void SaveAndClearImage(cv::Mat& image, const std::string& path)
    {
        try
        {
            cv::imwrite(path, image);
            LOG(INFO) << "PNG image file saved: " << path
                << ", dimensions: " << image.cols << "x" << image.rows;
        }
        catch (cv::Exception&)
        {
            LOG(ERROR) << "Error writing image file: " << path;
        }
        image.resize(0);
    }

    /**
    * @brief Saves the buffered raw float features into XML file and resets the image buffer
    *       line count to 0, so new lines are written from start again.
    */
    void SaveAndClearXML(cv::Mat& image, const std::string& path)
    {
        try
        {
            cv::FileStorage storage(path, cv::FileStorage::WRITE);
            storage << XML_MAT_IDENTIFIER << image;
            storage.release();
            LOG(INFO) << "XML file saved: " << path;
        }
        catch (cv::Exception&)
        {
            LOG(ERROR) << "Error writing XML file: " << path;
        }

        image.resize(0);
    }
};



#endif
