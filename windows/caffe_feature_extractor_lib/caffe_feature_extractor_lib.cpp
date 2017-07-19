#include "caffe_feature_extractor_lib.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace fs = boost::filesystem;


FeatureExtractor::FeatureExtractor(
    const string& modelFile,
    const string& trainedFile,
    const string& meanFile)
{
    LoadNetwork(modelFile, trainedFile);
    LoadMean(meanFile);
}


FeatureExtractor::~FeatureExtractor()
{
}

cv::Size FeatureExtractor::GetInputSize()
{
    return mInputGeometry;
}


void FeatureExtractor::ForwardImage(const Mat& image)
{
    //LOG(INFO)
    //    << "Passing image forward through the network...";

    if (image.empty())
    {
        LOG(ERROR) << "The input image is empty!";
        return;
    }
    else
    {
        vector<Mat> inputChannels;
        mNet->input_blobs()[0]->Reshape(1, mNumberOfChannels, mInputGeometry.height, mInputGeometry.width);
        mNet->Reshape();

        WrapInputLayer(&inputChannels);
        Preprocess(image, &inputChannels);
        mNet->Forward();

        //LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
    }
}


Mat FeatureExtractor::ExtractFeatures(const string& blobName, bool isKernelMaxPoolingEnabled)
{
    //LOG(INFO)
    //    << "Extracting feature from blob " << blobName;

    CHECK(mNet->has_blob(blobName))
        << "Unknown feature blob name: " << blobName;
    boost::shared_ptr<Blob<float>> blob = mNet->blob_by_name(blobName);

    Mat featureImage(1, blob->num() * blob->channels() * blob->width() * blob->height(),
        CV_32FC1, blob->data()->mutable_cpu_data());

    if (isKernelMaxPoolingEnabled)
    {
        // kernel max pooling
        Mat kernelMax(1, blob->num() * blob->channels(),
            CV_32FC1, blob->data()->mutable_cpu_data());
        
        int windowSize = blob->width() * blob->height();
        for (int iChannel = 0; iChannel < blob->channels(); iChannel++)
        {
            // find max
            float maximum = 0;
            for (int i = 0; i < windowSize; i++)
            {
                float value = featureImage.at<float>(0, (iChannel * windowSize) + i);
                if (value > maximum)
                {
                    maximum = value;
                }
            }

            // copy it
            kernelMax.at<float>(0, iChannel) = maximum;
        }
        return kernelMax;
    }
    else
    {
        // when returning as a vector:
        //const float* begin = (float*)blob->data()->cpu_data();
        //const float* end = begin + (blob->num() * blob->channels() * blob->width() * blob->height());
        //vector<float> feature(begin, end);

        return featureImage;
    }
}


Mat FeatureExtractor::ExtractFromImage(const Mat& image, const string& blobName)
{
    ForwardImage(image);
    return ExtractFeatures(blobName);
}


Mat FeatureExtractor::ExtractFromImages(const vector<Mat>& images, const string& blobName)
{
    vector<vector<Mat>> inputChannelsBatch;
    mNet->input_blobs()[0]->Reshape(images.size(), mNumberOfChannels, mInputGeometry.height, mInputGeometry.width);
    mNet->Reshape();

    WrapInputLayerBatch(&inputChannelsBatch);
    PreprocessBatch(images, &inputChannelsBatch);
    mNet->Forward();

    CHECK(mNet->has_blob(blobName))
        << "Unknown feature blob name: " << blobName;
    boost::shared_ptr<Blob<float>> blob = mNet->blob_by_name(blobName); 
    const float* begin = blob->cpu_data();
    const float* end = begin + blob->channels() * images.size();
    std::vector<float> outputBatch(begin, end);
    
    Mat featureImageBatch(blob->num(), blob->channels() * blob->width() * blob->height(),
        CV_32FC1, blob->data()->mutable_cpu_data());
    
    return featureImageBatch;
}


void FeatureExtractor::ExtractFromStream(
    const string& inputFolder,
    istream& inputStream,
    const string& outputPath,
    const string& blobNames,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    bool enableBinaryOutput,
    int imageMaxHeight,
    int logEveryNth,
    bool enableKernelMaxPooling)
{
    mIsTextOutputEnabled = enableTextOutput;
    mIsImageOutputEnabled = enableImageOutput;
    mIsXmlOutputEnabled = enableXmlOutput;
    mIsBinaryOutputEnabled = enableBinaryOutput;
    mImageMaxHeight = imageMaxHeight;
    mLogEveryNth = logEveryNth;
    mIsKernelMaxPoolingEnabled = enableKernelMaxPooling;

    LoadOutputModules(blobNames, outputPath);

    fs::path folder(inputFolder);
    fs::directory_iterator endIterator;

    CHECK(fs::exists(folder))
        << "Path not found: " << folder;

    CHECK(fs::is_directory(folder))
        << "Path is not directory: " << folder;

    double timeStart, timeElapsed;
    timeStart = (double)getTickCount();

    Mat image;
    int processedCount = 0;
    for (string file; getline(inputStream, file);)
    {
        string path = (folder / file).string();
        image = imread(path);
        if (image.empty())
        {
            LOG(ERROR) << "Unable to decode image " << path;
        }
        else
        {
            Process(image);

            for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
            {
                mOutputModules[iModule]->WriteFeatureFor(path);
            }

            LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
            processedCount++;
        }
    }

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Feature extraction finished in " << timeElapsed << " seconds.\n"
        << processedCount << " files processed.\n"
        << "Average processing speed is " << processedCount / timeElapsed << " features per second.";

    CloseOutputModules();
}


void FeatureExtractor::ExtractFromFileList(
    const string& inputFolder,
    const string& inputFile,
    const string& outputPath,
    const string& blobNames,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    bool enableBinaryOutput,
    int imageMaxHeight,
    int logEveryNth,
    bool enableKernelMaxPooling)
{
    ifstream inputStream(inputFile);
    if (inputStream.is_open())
    {
        ExtractFromStream(
            inputFolder,
            inputStream,
            outputPath,
            blobNames,
            enableTextOutput,
            enableImageOutput,
            enableXmlOutput,
            enableBinaryOutput,
            imageMaxHeight,
            logEveryNth,
            enableKernelMaxPooling);
    }
    else
    {
        LOG(FATAL) <<
            "Error opening file: " << inputFile;
    }
}



void FeatureExtractor::ExtractFromFileOrFolder(
    const string& inputFileOrFolder,
    const string& outputPath,
    const string& blobNames,
    bool enableTextOutput,
    bool enableImageOutput,
    bool enableXmlOutput,
    bool enableBinaryOutput,
    int imageMaxHeight,
    int logEveryNth,
    bool enableKernelMaxPooling)
{
    mIsTextOutputEnabled = enableTextOutput;
    mIsImageOutputEnabled = enableImageOutput;
    mIsXmlOutputEnabled = enableXmlOutput;
    mIsBinaryOutputEnabled = enableBinaryOutput;
    mImageMaxHeight = imageMaxHeight;
    mLogEveryNth = logEveryNth;
    mIsKernelMaxPoolingEnabled = enableKernelMaxPooling;

    LoadOutputModules(blobNames, outputPath);

    double timeStart, timeElapsed;
    LOG(INFO) << "Loading input directory...";
    timeStart = (double)getTickCount();

    fs::path inputPath(inputFileOrFolder);
    fs::directory_iterator endIterator;

    CHECK(fs::exists(inputPath))
        << "Path not found: " << inputPath;

    CHECK(fs::is_directory(inputPath) || fs::is_regular_file(inputPath))
        << "Path is not directory, nor a regular file: " << inputPath;

    Mat image;
    int processedCount = 0;
    if (fs::is_directory(inputPath))
    {
        for (fs::directory_iterator directoryIterator(inputPath); directoryIterator != endIterator; directoryIterator++)
        {
            if (fs::is_regular_file(directoryIterator->status()))
            {
                const string& file = directoryIterator->path().string();
                image = imread(file);
                if (image.empty())
                {
                    LOG(ERROR) << "Unable to decode image " << file;
                }
                else
                {
                    Process(image);

                    for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
                    {
                        mOutputModules[iModule]->WriteFeatureFor(file);
                    }

                    LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
                    processedCount++;
                }
            }
        }
    }
    else if (fs::is_regular_file(inputPath))
    {
        const string& file = inputPath.string();
        image = imread(file);
        if (image.empty())
        {
            LOG(ERROR) << "Unable to decode image " << file;
        }
        else
        {
            Process(image);

            for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
            {
                mOutputModules[iModule]->WriteFeatureFor(file);
            }

            LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
            processedCount++;
        }
    }

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Feature extraction finished in " << timeElapsed << " seconds.\n"
        << processedCount << " files processed.\n"
        << "Average processing speed is " << processedCount / timeElapsed << " features per second.";

    CloseOutputModules();
}


void FeatureExtractor::LoadNetwork(const string& modelFile, const string& trainedFile)
{
    double timeStart, timeElapsed;

    LOG(INFO) << "Loading network file...";
    timeStart = (double)getTickCount();

    mNet.reset(new Net<float>(modelFile, TEST));
    mNet->CopyTrainedLayersFrom(trainedFile);

    CHECK_EQ(mNet->num_inputs(), 1)
        << "Network should have exactly one input.";
    CHECK_EQ(mNet->num_outputs(), 1)
        << "Network should have exactly one output.";

    Blob<float>* inputLayer = mNet->input_blobs()[0];
    mNumberOfChannels = inputLayer->channels();
    CHECK(mNumberOfChannels == 3 || mNumberOfChannels == 1)
        << "Input layer should have 1 or 3 channels.";
    mInputGeometry = Size(inputLayer->width(), inputLayer->height());

    inputLayer->Reshape(1, mNumberOfChannels, mInputGeometry.height, mInputGeometry.width);
    /* Forward dimension change to all layers. */
    mNet->Reshape();

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Network file loaded in " << timeElapsed << " seconds.";
}


void FeatureExtractor::WrapInputLayer(vector<Mat>* inputChannels)
{
    Blob<float>* inputLayer = mNet->input_blobs()[0];

    int width = inputLayer->width();
    int height = inputLayer->height();
    float* inputData = inputLayer->mutable_cpu_data();
    for (int i = 0; i < inputLayer->channels(); ++i)
    {
        Mat channel(height, width, CV_32FC1, inputData);
        inputChannels->push_back(channel);
        inputData += width * height;
    }
}


void FeatureExtractor::WrapInputLayerBatch(vector<vector<Mat>> *inputChannelsBatch){
    Blob<float>* inputLayer = mNet->input_blobs()[0];

    int width = inputLayer->width();
    int height = inputLayer->height();
    int num = inputLayer->num();
    float* inputData = inputLayer->mutable_cpu_data();
    for (int j = 0; j < num; j++)
    {
        vector<cv::Mat> inputChannels;
        for (int i = 0; i < inputLayer->channels(); ++i)
        {
            cv::Mat channel(height, width, CV_32FC1, inputData);
            inputChannels.push_back(channel);
            inputData += width * height;
        }
        inputChannelsBatch->push_back(vector<cv::Mat>(inputChannels));
    }
}


void FeatureExtractor::LoadMean(const string& meanFile)
{
    double timeStart, timeElapsed;

    LOG(INFO) << "Loading mean file...";
    timeStart = (double)getTickCount();

    BlobProto blobProto;
    ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blobProto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> meanBlob;
    meanBlob.FromProto(blobProto);
    CHECK_EQ(meanBlob.channels(), mNumberOfChannels)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<Mat> channels;
    float* data = meanBlob.mutable_cpu_data();
    for (int i = 0; i < mNumberOfChannels; i++)
    {
        /* Extract an individual channel. */
        Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanBlob.height() * meanBlob.width();
    }

    /* Merge the separate channels into a single image. */
    merge(channels, mMean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    Scalar channelMean = cv::mean(mMean);
    mMean = Mat(mInputGeometry, mMean.type(), channelMean);

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Mean file loaded in " << timeElapsed << " seconds.";
}


void FeatureExtractor::LoadOutputModules(const string& blobNames, const string& outputPath)
{
    double timeStart, timeElapsed;

    LOG(INFO) << "Loading blob names...";
    timeStart = (double)getTickCount();

    vector<std::string> blobNamesSeparated;
    split(blobNamesSeparated, blobNames, boost::is_any_of(","));

    size_t numberOfFeatures = blobNamesSeparated.size();
    for (size_t i = 0; i < numberOfFeatures; i++)
    {
        mOutputModules.push_back(boost::make_shared<OutputModule>(mNet, fs::path(outputPath), blobNamesSeparated[i],
            mIsTextOutputEnabled, mIsImageOutputEnabled, mIsXmlOutputEnabled, mIsBinaryOutputEnabled, mImageMaxHeight, mIsKernelMaxPoolingEnabled));
    }

    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Blob names loaded in " << timeElapsed << " seconds. Loaded names are: " << blobNames;
}


void FeatureExtractor::Preprocess(const Mat& image, vector<Mat>* inputChannels)
{
    /* Convert the input image to the input image format of the network. */
    Mat sample;
    if (image.channels() == 3 && mNumberOfChannels == 1)
    {
        cvtColor(image, sample, CV_BGR2GRAY);
    }
    else if (image.channels() == 4 && mNumberOfChannels == 1)
    {
        cvtColor(image, sample, CV_BGRA2GRAY);
    }
    else if (image.channels() == 4 && mNumberOfChannels == 3)
    {
        cvtColor(image, sample, CV_BGRA2BGR);
    }
    else if (image.channels() == 1 && mNumberOfChannels == 3)
    {
        cvtColor(image, sample, CV_GRAY2BGR);
    }
    else
    {
        sample = image;
    }

    // TODO: beware, OpenCV resize leak
    Mat sampleResized;
    if (sample.size() != mInputGeometry)
    {
        resize(sample, sampleResized, mInputGeometry);
    }
    else
    {
        sampleResized = sample;
    }

    Mat sampleFloat;
    if (mNumberOfChannels == 3)
    {
        sampleResized.convertTo(sampleFloat, CV_32FC3);
    }
    else
    {
        sampleResized.convertTo(sampleFloat, CV_32FC1);
    }


    Mat sampleNormalized;
    subtract(sampleFloat, mMean, sampleNormalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the Mat
    * objects in input_channels. */
    split(sampleNormalized, *inputChannels);

    CHECK(reinterpret_cast<float*>(inputChannels->at(0).data) == mNet->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}


class ParallelPreprocess : public cv::ParallelLoopBody
{
private:
    const vector<Mat>& mImages;
    vector<vector<Mat>>* mInputChannelsBatch;
    int mNumberOfChannels;
    Size mInputGeometry;
    Mat mMean;

public:
    ParallelPreprocess(const vector<Mat>& images, vector<vector<Mat>>* inputChannelsBatch,
        int numberOfChannels, Size inputGeometry, Mat mean)
        : mImages(images), 
        mInputChannelsBatch(inputChannelsBatch), 
        mNumberOfChannels(numberOfChannels), 
        mInputGeometry(inputGeometry), 
        mMean(mean)
    {}

    virtual void operator()(const cv::Range &range) const 
    {
        for (int i = range.start; i != range.end; i++)
        {
            Mat image = mImages[i];
            vector<Mat> *inputChannels = &(mInputChannelsBatch->at(i));

            /* Convert the input image to the input image format of the network. */
            Mat sample;
            if (image.channels() == 3 && mNumberOfChannels == 1)
            {
                cvtColor(image, sample, CV_BGR2GRAY);
            }
            else if (image.channels() == 4 && mNumberOfChannels == 1)
            {
                cvtColor(image, sample, CV_BGRA2GRAY);
            }
            else if (image.channels() == 4 && mNumberOfChannels == 3)
            {
                cvtColor(image, sample, CV_BGRA2BGR);
            }
            else if (image.channels() == 1 && mNumberOfChannels == 3)
            {
                cvtColor(image, sample, CV_GRAY2BGR);
            }
            else
            {
                sample = image;
            }

            // TODO: beware, OpenCV resize leak
            Mat sampleResized;
            if (sample.size() != mInputGeometry)
            {
                resize(sample, sampleResized, mInputGeometry);
            }
            else
            {
                sampleResized = sample;
            }

            Mat sampleFloat;
            if (mNumberOfChannels == 3)
            {
                sampleResized.convertTo(sampleFloat, CV_32FC3);
            }
            else
            {
                sampleResized.convertTo(sampleFloat, CV_32FC1);
            }


            Mat sampleNormalized;
            subtract(sampleFloat, mMean, sampleNormalized);

            /* This operation will write the separate BGR planes directly to the
            * input layer of the network because it is wrapped by the Mat
            * objects in input_channels. */
            split(sampleNormalized, *inputChannels);
        }
    }
};


void FeatureExtractor::PreprocessBatch(const vector<Mat>& images, vector<vector<Mat>>* inputChannelsBatch)
{
    //cv::parallel_for_(cv::Range(0, images.size() - 1), 
    //    ParallelPreprocess(images, inputChannelsBatch, mNumberOfChannels, mInputGeometry, mMean));

    for (int i = 0; i < images.size(); i++)
    {
        Mat image = images[i];
        vector<Mat> *inputChannels = &(inputChannelsBatch->at(i));

        /* Convert the input image to the input image format of the network. */
        Mat sample;
        if (image.channels() == 3 && mNumberOfChannels == 1)
        {
            cvtColor(image, sample, CV_BGR2GRAY);
        }
        else if (image.channels() == 4 && mNumberOfChannels == 1)
        {
            cvtColor(image, sample, CV_BGRA2GRAY);
        }
        else if (image.channels() == 4 && mNumberOfChannels == 3)
        {
            cvtColor(image, sample, CV_BGRA2BGR);
        }
        else if (image.channels() == 1 && mNumberOfChannels == 3)
        {
            cvtColor(image, sample, CV_GRAY2BGR);
        }
        else
        {
            sample = image;
        }

        // TODO: beware, OpenCV resize leak
        Mat sampleResized;
        if (sample.size() != mInputGeometry)
        {
            resize(sample, sampleResized, mInputGeometry);
        }
        else
        {
            sampleResized = sample;
        }

        Mat sampleFloat;
        if (mNumberOfChannels == 3)
        {
            sampleResized.convertTo(sampleFloat, CV_32FC3);
        }
        else
        {
            sampleResized.convertTo(sampleFloat, CV_32FC1);
        }


        Mat sampleNormalized;
        subtract(sampleFloat, mMean, sampleNormalized);

        /* This operation will write the separate BGR planes directly to the
        * input layer of the network because it is wrapped by the Mat
        * objects in input_channels. */
        split(sampleNormalized, *inputChannels);

        //CHECK(reinterpret_cast<float*>(inputChannels->at(0).data) == mNet->input_blobs()[0]->cpu_data())
        //    << "Input channels are not wrapping the input layer of the network.";
    }
}



void FeatureExtractor::Process(const Mat& image)
{
    vector<Mat> inputChannels;
    WrapInputLayer(&inputChannels);
    Preprocess(image, &inputChannels);
    mNet->Forward();
}


void FeatureExtractor::CloseOutputModules()
{
    double timeStart, timeElapsed;
    // close output modules
    LOG(INFO) << "Closing output modules...";
    timeStart = (double)getTickCount();
    for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
    {
        mOutputModules[iModule]->Close();
    }
    timeElapsed = ((double)getTickCount() - timeStart) / getTickFrequency();
    LOG(INFO) << "Output modules closed in " << timeElapsed << " seconds.\n";
}
