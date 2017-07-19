#include "../caffe_feature_extractor_lib/caffe_feature_extractor_lib.hpp"

using namespace std;
namespace fs = boost::filesystem;

void PrintHelpMessage()
{
    std::cerr <<
        "Caffe feature extractor\nAuthor: Gregor Kovalcik\n\n"
        "This program takes in a trained network and an input data layer name, and then\n"
        "extracts features of the input data produced by the net.\n\n"
        "Usage:\n"
        "    caffe_feature_extractor_cli [options] deploy.prototxt network.caffemodel\n"
        "    mean.binaryproto blob_name1[,name2,...] input_folder file_list\n"
        "    output_file_or_folder\n"
        "Note:\n"
        "    you can extract multiple features in one pass by specifying multiple\n"
        "    feature blob names separated by ','. The names cannot contain white space\n"
        "    characters.\n\n"

        "Options:\n"

        "-h, --help\n"
        "    Shows the help screen.\n\n"

#ifndef CPU_ONLY
        "-m <GPU|CPU>, --mode <GPU|CPU> (default: GPU)\n"
        "    Choose whether to compute features using GPU or CPU.\n\n"
#endif

        "-d, --disable-text-output\n"
        "    Disables the text file output (useful to generate image file output only).\n\n"

        "-i, --image-output\n"
        "    Enables the image output. Each row is the feature of one input image.\n"
        "    Number of columns is equal to the extracted blob size.\n"
        "    This generates four PNG image files per extracted blob. The original\n"
        "    filename is preserved and the extension is replaced:\n"
        "    - (output_filename).png:\n"
        "        Grayscale image, where the feature values are normalized to range 0..1\n"
        "    - (output_filename)_hc.png:\n"
        "        High contrast version of the normalized image. Zero values are copied,\n"
        "        positive values are set to 1 (255 actually, because we are saving it\n"
        "        using 8bits per pixel).\n"
        "    - (output_filename)_br.png:\n"
        "        RGB image, where the feature values are normalized to range -1..+1.\n"
        "        Negative values are printed in blue color while positive values are\n"
        "        printed in red color.\n"
        "    - (output_filename)_brhc.png:\n"
        "        High contrast version of the previous image. Zero values are copied,\n"
        "        positive values are set to red color RGB(255, 0, 0), negative values\n"
        "        are set to blue color RGB(0, 0, 255).\n\n"

        "-r <(int) height>, --image-height <(int) height> (default: 0 - do not split)\n"
        "    Splits the image files if they are higher than the <(int) height>. Useful\n"
        "    when the generated images are too big to fit in the memory.\n\n"

        "-x, --xml-output\n"
        "    Enables XML output. Stores features as OpenCV CV_32FC1 Mat. Each row is\n"
        "    the feature of one input image. This Mat is then serialized using\n"
        "    cv::FileStorage with identifier \"caffe_features\".\n\n"

        "-l <(int) log_level>, --log-level <(int) log_level> (default: 0)\n"
        "    Log suppression level: messages logged at a lower level than this are.\n"
        "    suppressed. The numbers of severity levels INFO, WARNING, ERROR, and FATAL\n"
        "    are 0, 1, 2, and 3, respectively.\n\n"

        "-n <(int) log_level>, --log-every-nth <(int) log_level> \n"
        "    (default in GPU mode: 100, default in CPU mode: 10)\n"
        "    Logs every nth file processed."

        << std::endl;
}


string ParseArgumentValueForOption(const string& option, char * const * const& iterator, char * const * const& end)
{
    CHECK(iterator != end) << "Error parsing option \"" << option << "\"";
    return *iterator;
}


string modelFile;
string trainedFile;
string meanFile;
string blobNames;
string inputFolder;
string inputFileList;
string outputPath;

bool isTextOutputEnabled = true;
bool isImageOutputEnabled = false;
bool isXmlOutputEnabled = false;
bool isBinaryOutputEnabled = false;
int imageMaxHeight = 0;
bool isKernelMaxPoolingEnabled = false;

#ifdef CPU_ONLY
int logEveryNth = 10;
#else
int logEveryNth = 100;
#endif


bool ParseProgramArguments(int argc, char * const * const argv)
{
    char * const * const begin = argv;
    char * const * const end = argv + argc;
    char * const * iterator = begin + 1;	// skip the first argument

    // no arguments
    if (iterator == end)
    {
        PrintHelpMessage();
        return false;
    }

    // initialize logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 0;

#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif


    // parse optional arguments
    while (iterator != end && **iterator == '-')
    {
        string option = *iterator;
        string value = "";

        if (!option.compare("-h") || !option.compare("--help"))				// help
        {
            PrintHelpMessage();
            return false;
        }
#ifndef CPU_ONLY
        else if (!option.compare("-m") || !option.compare("--mode"))			// CPU/GPU mode
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            boost::algorithm::to_upper(value);
            if (value == "GPU")
            {
                caffe::Caffe::set_mode(caffe::Caffe::GPU);
            }
            else if (value == "CPU")
            {
                caffe::Caffe::set_mode(caffe::Caffe::CPU);
                logEveryNth = 10;
            }
            else
            {
                cerr << "Unknown mode: " << value << endl;
                return false;
            }
        }
#endif
        else if (!option.compare("-d") || !option.compare("--disable-text-output"))	// text output
        {
            isTextOutputEnabled = false;
        }
        else if (!option.compare("-i") || !option.compare("--image-output"))			// image output
        {
            isImageOutputEnabled = true;
        }
        else if (!option.compare("-r") || !option.compare("--image-height"))			// image height
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            try
            {
                imageMaxHeight = std::stoi(value);
            }
            catch (const invalid_argument&)
            {
                cerr << "Error parsing image height:" << value << endl;
                return false;
            }
        }
        else if (!option.compare("-x") || !option.compare("--xml-output"))			// XML output
        {
            isXmlOutputEnabled = true;
        }
        else if (!option.compare("-b") || !option.compare("--binary-output"))		// binary output
        {
            isBinaryOutputEnabled = true;
        }
        else if (!option.compare("-l") || !option.compare("--log-level"))			// log level
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            int logLevel;
            try
            {
                logLevel = std::stoi(value);
            }
            catch (const invalid_argument&)
            {
                cerr << "Error parsing log level!" << endl;
                return false;
            }
            FLAGS_minloglevel = logLevel;
        }
        else if (!option.compare("-n") || !option.compare("--log-every-nth"))		// log every nth
        {
            value = ParseArgumentValueForOption(option, ++iterator, end);
            try
            {
                logEveryNth = std::stoi(value);
            }
            catch (const invalid_argument&)
            {
                cerr << "Error parsing \"log every n-th\":" << value << endl;
                return false;
            }
        }
        else if (!option.compare("-k") || !option.compare("--kernel-max-pooling"))	// Kernel max pooling
        {
            isKernelMaxPoolingEnabled = true;
        }
        else
        {
            cerr << "Unknown option: " << option;
            return false;
        }

        iterator++;
    }

    // parse mandatory arguments
    modelFile = ParseArgumentValueForOption("modelFile", iterator++, end);
    trainedFile = ParseArgumentValueForOption("trainedFile", iterator++, end);
    meanFile = ParseArgumentValueForOption("meanFile", iterator++, end);
    blobNames = ParseArgumentValueForOption("blobNames", iterator++, end);
    inputFolder = ParseArgumentValueForOption("inputFolder", iterator++, end);
    inputFileList = ParseArgumentValueForOption("inputFileList", iterator++, end);
    outputPath = ParseArgumentValueForOption("outputPath", iterator++, end);

    return true;
}


int main(int argc, char** argv)
{
    if (ParseProgramArguments(argc, argv))
    {
        FeatureExtractor featureExtractor(modelFile, trainedFile, meanFile);

        featureExtractor.ExtractFromFileList(inputFolder, inputFileList, outputPath, blobNames,
            isTextOutputEnabled, isImageOutputEnabled, isXmlOutputEnabled, isBinaryOutputEnabled, imageMaxHeight, logEveryNth, isKernelMaxPoolingEnabled);
        
        //int batchSize = 12; // VGG/2
        ////int batchSize = 5; // ResNet152
        ////int batchSize = 400; // AlexNet
        //int processedCount = 0;
        //fs::directory_iterator endIterator;
        //if (fs::is_directory(inputFolder))
        //{
        //    for (fs::directory_iterator subdirectoryIterator(inputFolder); subdirectoryIterator != endIterator; subdirectoryIterator++)
        //    {
        //        if (!fs::is_directory(subdirectoryIterator->status())) continue;
        //        
        //        fs::path outputFilePath = (fs::path(outputPath) / fs::path(subdirectoryIterator->path().filename().string() + ".xml"));
        //        if (fs::exists(outputFilePath))
        //        {
        //            LOG(INFO) << "Skipped: " << subdirectoryIterator->path().filename();
        //            continue;
        //        }


        //        vector<fs::path> filenames;
        //        for (fs::directory_iterator fileIterator(subdirectoryIterator->path()); fileIterator != endIterator; fileIterator++)
        //        {

        //            if (fs::is_regular_file(fileIterator->status()))
        //            {
        //                filenames.push_back(fileIterator->path());
        //            }
        //        }
        //        sort(filenames.begin(), filenames.end());
        //        //const string& file = directoryIterator->path().string();
        //        //image = imread(file);
        //        //if (image.empty())
        //        //{
        //        //    LOG(ERROR) << "Unable to decode image " << file;
        //        //}
        //        //else
        //        //{
        //        //    Process(image);

        //        //    for (int iModule = 0; iModule < ((int)mOutputModules.size()); iModule++)
        //        //    {
        //        //        mOutputModules[iModule]->WriteFeatureFor(file);
        //        //    }

        //        //    LOG_EVERY_N(INFO, mLogEveryNth) << google::COUNTER << " processed.";
        //        //    processedCount++;
        //        //}

        //        vector<cv::Mat> images;
        //        cv::Mat result;
        //        for (size_t i = 0; i < filenames.size(); i += batchSize)
        //        {
        //            images.clear();
        //            for (size_t j = 0; j < batchSize && j + i < filenames.size(); j++)
        //            {
        //                string imagePath = filenames[i + j].string();
        //                cv::Mat image = cv::imread(imagePath);
        //                if (image.empty())
        //                {
        //                    cerr << "Unable to read image: " << filenames[i + j].string() << endl;
        //                }
        //                images.push_back(image);
        //            }

        //            cv::Mat features = featureExtractor.ExtractFromImages(images, blobNames);
        //            
        //            //cv::Mat kernelMax(features.rows, 256, features.type());
        //            //for (int iRow = 0; iRow < features.rows; iRow++)
        //            //{
        //            //    for (int j = 0; j < 256; j++)
        //            //    {
        //            //        int offset = j * 13 * 13;
        //            //        float maximum = features.at<float>(iRow, offset);
        //            //        for (int k = offset + 1; k < 13 * 13; k++)
        //            //        {
        //            //            maximum = features.at<float>(iRow, k) > maximum ? features.at<float>(iRow, k) : maximum;
        //            //        }
        //            //        kernelMax.at<float>(iRow, j) = maximum;
        //            //    }
        //            //}
        //            //result.push_back(kernelMax);

        //            result.push_back(features);
        //            //cout << i + batchSize << " processed." << endl;
        //        }

        //        try
        //        {
        //            cv::FileStorage storage(outputFilePath.string(), cv::FileStorage::WRITE);
        //            storage << XML_MAT_IDENTIFIER << result;
        //            storage.release();
        //            LOG(INFO) << "XML file saved: " << outputFilePath.string();
        //        }
        //        catch (cv::Exception&)
        //        {
        //            LOG(ERROR) << "Error writing XML file: " << outputFilePath.string();
        //        }

        //        //path = (fs::path(inputFolder) / fs::path(subdirectoryIterator->path().filename().string() + ".png")).string();
        //        //double min, max;
        //        //minMaxLoc(result, &min, &max);
        //        //min = (min > 0) ? 0 : min;
        //        //result += abs(min);
        //        //result *= 1.0 / abs(max - min);
        //        //result.convertTo(result, CV_8U, 255);
        //        //cv::imwrite(path, result);
        //    }
        //}
        //else
        //{
        //    cerr << "Error: not a directory: " << inputFolder << endl;
        //    return 1;
        //}

        //vector<cv::String> filenames;
        //vector<cv::Mat> images;
        //cv::Mat resultAll;
        //cv::glob(inputFolder, filenames, true); // recurse
        //int batchSize = 400;
        //for (size_t i = 0; i < filenames.size() / batchSize; i++)
        //{
        //    images.clear();
        //    for (size_t j = 0; j < batchSize && j + i * batchSize < filenames.size(); j++)
        //    {
        //        cv::Mat image = cv::imread(filenames[i]);
        //        if (image.empty()) continue; //only proceed if sucsessful
        //        // you probably want to do some preprocessing
        //        images.push_back(image);
        //    }

        //    cv::Mat result = featureExtractor.ExtractFromImages(images, blobNames);
        //    resultAll.push_back(result);
        //    cout << i * batchSize << endl;
        //}
    }
    else
    {
        return 1;
    }

    return 0;
}
