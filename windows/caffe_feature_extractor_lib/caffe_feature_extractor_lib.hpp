#ifndef CAFFE_FEATURE_EXTRACTOR_LIB_HPP
#define CAFFE_FEATURE_EXTRACTOR_LIB_HPP

#include <caffe/caffe.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/make_shared.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "output_module.hpp"

/**
* @brief Implementation of caffe feature extractor. The features are extracted from
*       the selected output blobs of the network layers.
*       The feature can be either returned as a cv:Mat or written directly
*       to the hard drive using the OutputModule class.
*/
class FeatureExtractor
{
public:
    /**
    * @brief Feature extractor constructor.
    * @param modelFile model file (topological definition of the network).
    * @param trainedFile trained file (trained neuron weights).
    * @param meanFile mean file (mean image of the database the model was trained on).
    */
    FeatureExtractor(
        const std::string& modelFile,
        const std::string& trainedFile,
        const std::string& meanFile);

    ~FeatureExtractor();

    /**
    * @brief Returns the dimensions of the input image
    */
    cv::Size GetInputSize();


    /**
    * @brief Passes forward an image throught the network.
    * @param image Input image of CV_8U type.
    */
    void ForwardImage(const cv::Mat& image);


    /**
    * @brief Extracts features from a blob.
    * @param blobName Name of the blob to extract the features from.
    * @returns cv::Mat of type CV_32F and dimensions 1x[feature_size] - one feature per column.
    */
    cv::Mat ExtractFeatures(const std::string& blobName, bool isKernelMaxPoolingEnabled = false);


    /**
    * @brief Extracts features from a blob after forwarding the input image
    *       through the preloaded network.
    * @param image Input image of CV_8U type.
    * @param blobName Name of the blob to extract the features from.
    * @returns cv::Mat of type CV_32F and dimensions 1x[feature_size] - one feature per column.
    */
    cv::Mat ExtractFromImage(const cv::Mat& image, const std::string& blobName);


    /**
    * @brief Extracts features from a blob after forwarding the input images
    *       through the preloaded network.
    * @param images Input images of CV_8U type.
    * @param blobName Name of the blob to extract the features from.
    * @returns vector of cv::Mat of type CV_32F and dimensions 1x[feature_size] - one feature per column.
    */
    cv::Mat ExtractFromImages(const std::vector<cv::Mat>& images, const std::string& blobName);


    /**
    * @brief Extracts features from image files in a directory, whose filenames are read from the input stream,
    *       one filename per line.
    * @param inputFolder Input path to the folder containing input files.
    * @param inputStream Input stream containing image filenames.
    * @param outputStream Output stream to write features to if the text output is enabled.
    * @param blobNames Names of blobs to extract, delimited by comma ("blob1,blob2,blob3").
    * @param logEveryNth How often should be the info log printed to the console.
    */
    void ExtractFromStream(
        const std::string& inputFolder,
        std::istream& inputStream,
        std::ostream& outputStream,
        const std::string& blobNames,
        int logEveryNth = 100);

    /**
    * @brief Extracts features from image files in a directory, whose filenames are read from the input stream,
    *       one filename per line.
    * @param inputFolder Input path to the folder containing input files.
    * @param inputStream Input stream containing image filenames.
    * @param outputPath Output path. The filename will be appended by the blob name identifier "_blobName".
    * @param blobNames Names of blobs to extract, delimited by comma ("blob1,blob2,blob3").
    * @param enableTextOutput Whether to write features to the output stream.
    * @param enableImageOutput Whether to visualize features in an image file.
    * @param enableXmlOutput Whether to serialize features as a cv::Mat in a XML file using cv::FileStorage.
    * @param imageMaxHeight Maximal height of image and XML output files. The image and XML outputs are stored
    *       in the memory until the output files are written, so for large databases it may be useful
    *       to split the output into multiple files.
    * @param logEveryNth How often should be the info log printed to the console.
    */
    void ExtractFromStream(
        const std::string& inputFolder,
        std::istream& inputStream,
        const std::string& outputPath,
        const std::string& blobNames,
        bool enableTextOutput = true,
        bool enableImageOutput = false,
        bool enableXmlOutput = false,
        bool enableBinaryOutput = false,
        int imageMaxHeight = 0,
        int logEveryNth = 100,
        bool enableKernelMaxPooling = false);

    /**
    * @brief Extracts features from the filenames specified in the input file (one filename per line).
    * @param inputFolder Input path to a file or folder.
    * @param inputFile Input stream containing image filenames.
    * @param outputPath Output path. The filename will be appended by the blob name identifier "_blobName".
    * @param blobNames Names of blobs to extract, delimited by comma ("blob1,blob2,blob3").
    * @param enableTextOutput Whether to write features to the output stream.
    * @param enableImageOutput Whether to visualize features in an image file.
    * @param enableXmlOutput Whether to serialize features as a cv::Mat in a XML file using cv::FileStorage.
    * @param imageMaxHeight Maximal height of image and XML output files. The image and XML outputs are stored
    *       in the memory until the output files are written, so for large databases it may be useful
    *       to split the output into multiple files.
    * @param logEveryNth How often should be the info log printed to the console.
    */
    void ExtractFromFileList(
        const std::string& inputFolder,
        const std::string& inputFile,
        const std::string& outputPath,
        const std::string& blobNames,
        bool enableTextOutput = true,
        bool enableImageOutput = false,
        bool enableXmlOutput = false,
        bool enableBinaryOutput = false,
        int imageMaxHeight = 0,
        int logEveryNth = 100,
        bool enableKernelMaxPooling = false);


    /**
    * @brief Extracts features from the input path.
    *       If the path points to an image file, only this one file is processed.
    *       If the path points to a directory, all supported image files in this directory are processed.
    *       The unsupported files are skipped and a warning is written to the console.
    * @param inputFileOrFolder Input path to a file or folder.
    * @param outputPath Output path. The filename will be appended by the blob name identifier "_blobName".
    * @param blobNames Names of blobs to extract, delimited by comma ("blob1,blob2,blob3").
    * @param enableTextOutput Whether to write features to the output stream.
    * @param enableImageOutput Whether to visualize features in an image file.
    * @param enableXmlOutput Whether to serialize features as a cv::Mat in a XML file using cv::FileStorage.
    * @param imageMaxHeight Maximal height of image and XML output files. The image and XML outputs are stored
    *       in the memory until the output files are written, so for large databases it may be useful
    *       to split the output into multiple files.
    * @param logEveryNth How often should be the info log printed to the console.
    */
    void ExtractFromFileOrFolder(
        const std::string& inputFileOrFolder,
        const std::string& outputPath,
        const std::string& blobNames,
        bool enableTextOutput = true,
        bool enableImageOutput = false,
        bool enableXmlOutput = false,
        bool enableBinaryOutput = false,
        int imageMaxHeight = 0,
        int logEveryNth = 100,
        bool enableKernelMaxPooling = false);


    void EnableTextOutput()
    {
        mIsTextOutputEnabled = true;
    }
    void DisableTextOutput()
    {
        mIsTextOutputEnabled = false;
    }
    bool IsTextOutputEnabled()
    {
        return mIsTextOutputEnabled;
    }

    void EnableBinaryOutput()
    {
        mIsBinaryOutputEnabled = true;
    }
    void DisableBinaryOutput()
    {
        mIsBinaryOutputEnabled = false;
    }
    bool IsBinaryOutputEnabled()
    {
        return mIsBinaryOutputEnabled;
    }

    void EnableImageOutput()
    {
        mIsImageOutputEnabled = true;
    }
    void DisableImageOutput()
    {
        mIsImageOutputEnabled = false;
    }
    bool IsImageOutputEnabled()
    {
        return mIsImageOutputEnabled;
    }

    void EnableXmlOutput()
    {
        mIsXmlOutputEnabled = true;
    }
    void DisableXmlOutput()
    {
        mIsXmlOutputEnabled = false;
    }
    bool IsXmlOutputEnabled()
    {
        return mIsXmlOutputEnabled;
    }

    void SetImageMaxHeight(int maxHeight)
    {
        mImageMaxHeight = maxHeight;
    }
    int GetImageMaxHeight()
    {
        return mImageMaxHeight;
    }

    void SetLogEveryNth(int logEveryNth)
    {
        mLogEveryNth = logEveryNth;
    }
    int GetLogEveryNth()
    {
        return mLogEveryNth;
    }

    bool IsKernelMaxPoolingEnabled()
    {
        return mIsKernelMaxPoolingEnabled;
    }

private:
    bool mIsTextOutputEnabled;
    bool mIsImageOutputEnabled;
    bool mIsXmlOutputEnabled;
    bool mIsBinaryOutputEnabled;
    int mImageMaxHeight;	    // output images are split into multiple files to fit this height
    int mLogEveryNth;

    bool mIsKernelMaxPoolingEnabled;    // compute maximal activation over all results of one kernel
                                        // resulting feature dimension is equal to blob channel dimension
                                        // (number of layer kernels)
public:
    boost::shared_ptr<caffe::Net<float>> mNet;	// Caffe net used to generate and extract features from
private:
    cv::Size mInputGeometry;					// of the first layer of the network
    int mNumberOfChannels;						// of the first layer of the network
    cv::Mat mMean;
    //std::vector<std::string> mInputFiles;		// preloaded filenames of files to extract features from
    std::vector<boost::shared_ptr<OutputModule>> mOutputModules;	// used to write extracted features to disk in multiple formats


    /**
    * @brief Loads the network using model and trained file (network model and weights of its neurons).
    * @param modelFile model file (topological definition of the network).
    * @param trainedFile trained file (trained neuron weights).
    */
    void LoadNetwork(const std::string& modelFile, const std::string& trainedFile);

    /*
    * @brief Wrap the input layer of the network in separate Mat objects
    * (one per channel). This way we save one memcpy operation and we
    * don't need to rely on cudaMemcpy2D. The last preprocessing
    * operation will write the separate channels directly to the input
    * layer.
    */
    void WrapInputLayer(std::vector<cv::Mat>* inputChannels);


    /*
    * @brief Wrap the input layer of the network in separate Mat objects
    * (one per channel). This way we save one memcpy operation and we
    * don't need to rely on cudaMemcpy2D. The last preprocessing
    * operation will write the separate channels directly to the input
    * layer.
    */
    void WrapInputLayerBatch(std::vector<std::vector<cv::Mat>>* inputChannelsBatch);

    /*
    * @brief Load the mean file in binaryproto format.
    * @param meanFile mean file (mean image of the database the model was trained on).
    */
    void LoadMean(const std::string& meanFile);

    /*
    * @brief Parse blob names and create an output module for each blob.
    * @param blobNames Names of blobs to extract, delimited by comma ("blob1,blob2,blob3").
    * @param outputPath Output path. The filename will be appended by the blob name identifier "_blobName".
    *       Also the output module appends the identifier of an output method.
    */
    void LoadOutputModules(const std::string& blobNames, const std::string& outputPath);

    /*
    * @brief Resize, convert to the correct image format and write to the first layer of the network.
    * @param image Image to be preprocessed and written to the first layer of the network.
    * @param inputChannels cv::Mat wrapping memory data of the first network layer.
    *       Writing to this cv::Mat, one will write to the first layer directly.
    */
    void Preprocess(const cv::Mat& image, std::vector<cv::Mat>* inputChannels);


    /*
    * @brief Resize, convert to the correct image format and write to the first layer of the network.
    * @param images Images to be preprocessed and written to the first layer of the network.
    * @param inputChannelsBatch cv::Mat wrapping memory data of the first network layer.
    *       Writing to this cv::Mat, one will write to the first layer directly.
    */
    void PreprocessBatch(const std::vector<cv::Mat>& images, std::vector<std::vector<cv::Mat>>* inputChannelsBatch);


    /*
    * @brief Processes the image.
    * @param image Image to be processed.
    */
    void Process(const cv::Mat& image);

    /*
    * @brief Closes all output modules, saves images stored in buffers.
    */
    void CloseOutputModules();
};


#endif
