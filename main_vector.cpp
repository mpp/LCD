#include <fstream>
#include <locale>
#include <iomanip>
#include <limits>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

#include "frameposelist.h"
#include "matchviewer.h"
#include "BowMatchers/vectorbowmatcher.h"
#include "MOSAIC/mosaic.h"

int help(void);
void groundTruthOverlay(const cv::Mat &inputMat, cv::Mat &outputMat);

double computeRotationDifference(const cv::Vec3f R1, const cv::Vec3f R2);

int main(int argc, char **argv) {
    
    /////////////////////////////    
    /// Check arguments
    
    if (argc != 5)
    {
        help();
        exit(-1);
    }
    
    if (std::string(argv[1]) != "-s" || std::string(argv[3]) != "-l")
    {
        help();
        exit(-1);
    }
    
    std::cout << std::fixed << std::setprecision(6) << "Hello!" << std::endl;
    setlocale(LC_NUMERIC, "C");
    
    /////////////////////////////    
    /// Open input files
    
    std::string 
        logFileName = argv[4];
    
    std::ifstream 
        ifsLog(logFileName.c_str());
    
    if (!ifsLog)
    {
        std::cerr << "Cannot open file" << logFileName << std::endl;
        exit(-1);
    }
    
    std::string 
        settfilename = argv[2];
    
    cv::FileStorage 
        fs;
    
    fs.open(settfilename, cv::FileStorage::READ);
    
    if (!fs.isOpened()) 
    {
        std::cerr << "Could not open settings file: " << settfilename << std::endl;
        exit(-1);
    }
    
    /////////////////////////////    
    /// Setup the Flann Bow Matcher
    
    LCD::BOWExtractor
        bowExtractor(fs);
    
    cv::Mat
        actualFrame,
        similarFrame,
        bow;
    
    std::vector<cv::KeyPoint>
        kpts;
    
    cv::Mat 
        completeDescriptors;
    
    std::vector< cv::DMatch >
        matches;
    
    int 
        similarFrameMapIndex = -1,
        bestScore = -1;
    
    double
        minDistance = std::numeric_limits< double >::max();
    
    std::vector< std::vector < int > > 
        pointIDXOfCLusters;
    
    /////////////////////////////    
    /// Setup the FramePoseList
    
    LCD::FramePoseList 
        list(fs);
    
    std::string 
        temp,
        type,
        frameName;
    
    unsigned int
        timestamp;
    
    cv::Vec3f
        poseTranslation,
        poseRotation,
        tA = cv::Vec3f(0,0,0), 
        tB = cv::Vec3f(0,0,0),
        rA = cv::Vec3f(0,0,0), 
        rB = cv::Vec3f(0,0,0);
        
    cv::Mat
        frameA, frameB, Atmp, Btmp;
        
    std::string referenceFrameName;
    
    bool 
        associateFramePos = false,
        frameAtaken = false;
    
    /////////////////////////////    
    /// Setup the viewer class
    
    LCD::MatchViewer 
        matchviewer;
    
    cv::namedWindow("Window");
    
    cv::Mat
        window,
        actualFrameBGR,
        similarFrameBGR;
    
    std::string 
        basePath = fs["FilePaths"]["TestImagesBasePath"];
    
    /////////////////////////////    
    /// Frame selection variables
    
    float
        distanceThreshold = fs["loopDetectorOptions"]["distanceThreshold"];
    
    int
        corrispondenceThreshold = 0,
        corrispondenceCounter = 0;
        
    fs["loopDetectorOptions"]["scoreThreshold"] >> corrispondenceThreshold;
    
    bool
        corrispondenceThresholdConstraint = false;
        
    ///TODO: move parallaxDistanceThreshold into the settings file
    double  
        parallaxDistanceThreshold = fs["loopDetectorOptions"]["parallaxDistance"];
    
    /////////////////////////////    
    /// Main loop
    
    // Loop the log file
    std::cout << std::endl << "-------" << std::endl << "Looping: " << logFileName << std::endl << "-------" << std::endl;
    
    // Debug variables
    int 
        frameCount = 0,
        mapCount = 0,
        kkk = 0;
        
    while (!ifsLog.eof())
    {
        // First part of each row is TIME: <timestamp>
        ifsLog >> temp;
        ifsLog >> temp;
        ifsLog >> timestamp;
        
        //     std::cout << time << " ";
        
        // Second part of each row start with the type, that can be: POS, IMU, IMAGE
        ifsLog >> type;
        ifsLog >> temp;
        
// FIRST TRANCHE
// first: 0
// last: 68659126
// SECOND TRANCHE
// first: 68659126
// last: 140689126
// THIRD TRANCHE
// first: 140689126
// last: 208485940
// FOURTH TRANCHE
// first: 208485940
// last: 281215981
        if (timestamp > 281215981)
        {
            break;
        }
        if (timestamp <= 208485940)
        {
            if (kkk % 1000 == 0)
            {
                std::cout << kkk << timestamp << std::endl;
            }
            kkk++;
            
            if (type.compare("POS") == 0)
            {
                ifsLog >> poseTranslation(0) 
                >> poseTranslation(1) 
                >> poseTranslation(2) 
                >> poseRotation(0)
                >> poseRotation(1)
                >> poseRotation(2);
            }
            else if (type.compare("IMAGE") == 0)
            {
                ifsLog >> frameName;
            }
            else if (type.compare("IMU") == 0)
            {
                // Ignore IMU data
                ifsLog >> temp >> temp >> temp >> temp >> temp >> temp;
            }
            
            continue;
        }
        
        if (type.compare("POS") == 0)
        {
            ifsLog >> poseTranslation(0) 
                >> poseTranslation(1) 
                >> poseTranslation(2) 
                >> poseRotation(0)
                >> poseRotation(1)
                >> poseRotation(2);
            
            if (associateFramePos)
            {
                if (!frameAtaken)
                {
                    double distance = cv::norm(tA - poseTranslation);
                    
                    if(distance >= distanceThreshold)
                    {
                        // take frame A
                        frameAtaken = true;
                        
                        tA = poseTranslation;
                        rA = poseRotation;
                        
                        frameA = cv::imread(basePath + frameName, CV_LOAD_IMAGE_GRAYSCALE);
                        
                        referenceFrameName = frameName;
                    }
                }
                else
                {
                    double parallaxDistance = cv::norm(tA - poseTranslation);
                    
                    double bearingDifference = computeRotationDifference(rA, rB);
                    
                    // take frame B checking the distance and the orientation...
                    // Maybe I should take care of the rotation, it should be the same of frame A
                    double bearingDifferenceThreshold = 0.1;
                    if (parallaxDistance >= parallaxDistanceThreshold && bearingDifference <= bearingDifferenceThreshold)
                    {
                        std::cout << "Extracting bow from reference frame: " << referenceFrameName << " with support frame: " << frameName << std::endl;
                        
                        tB = poseTranslation;
                        rB = poseRotation;
                        
                        frameB = cv::imread(basePath + frameName, CV_LOAD_IMAGE_GRAYSCALE);
                        
                        /// Pass it to the FLANN BoW Matcher
                        kpts.clear();
                        pointIDXOfCLusters.clear();
                        
                        // Initialize the descriptor extractor (compute optimized normals)
                        MOSAIC::MOSAIC mo(fs, frameA, frameB, tA, tB, rA, rB, referenceFrameName);
                        
                        // Extract the descriptors
                        cv::Mat descriptorsBOW;
                        std::vector<cv::Vec3d> triangulatedPoints;
                        mo.computeDescriptors(kpts, descriptorsBOW, completeDescriptors, triangulatedPoints);
                        
                        
                        if (kpts.size() < corrispondenceThreshold)
                        {
                            std::cout << "main said: Bad couple of images." << std::endl;
                            
                            // Use this as reference frame
                            tA = tB;
                            rA = rB;
                            referenceFrameName = frameName;
                            
                            frameA = frameB.clone();
                            
                            continue;
                        }
                        
                        bowExtractor.compute(descriptorsBOW, bow, pointIDXOfCLusters);
                        
                        cv::FileStorage outputFile("descriptors/"+referenceFrameName+".yml", cv::FileStorage::WRITE);
                        
                        outputFile << "ReferenceFrame" << referenceFrameName;
                        outputFile << "SupportFrame" << frameName;
                        outputFile << "tA" << tA;
                        outputFile << "tB" << tB;
                        outputFile << "rA" << rA;
                        outputFile << "rB" << rB;
                        outputFile << "BOW" << bow;
                        outputFile << "Descriptors" << completeDescriptors;
                        outputFile << "KPTS" << kpts;
                        outputFile << "IDX" << pointIDXOfCLusters;
                        outputFile << "TriangulatedPoints" << triangulatedPoints;
                        
                        outputFile.release();
                        
                        // reset the flag
                        frameAtaken = false;
                    }
                }
                // reset flag
                associateFramePos = false;
            }
            else 
            {
                list.add("", poseTranslation, poseTranslation, timestamp);
            }
        }
        else if (type.compare("IMAGE") == 0)
        {
            ifsLog >> frameName;
            
            // Associate next POS with this frame
            associateFramePos = true; 
            
            frameCount++;
            //             std::cout << "Frame #:" << frameCount << std::endl;
        }
        else if (type.compare("IMU") == 0)
        {
            // Ignore IMU data
            ifsLog >> temp >> temp >> temp >> temp >> temp >> temp;
        }
    }
    
    std::cout << "Looping end. See you." << std::endl;  
    
    
    return 0;
}


/** Add the specific ground truth overlay for the indoor loop dataset
 */
void groundTruthOverlay(const cv::Mat &inputMat, cv::Mat &outputMat)
{
    cv::cvtColor(inputMat, outputMat, CV_GRAY2BGR);
    
    // Iterate the matrix rows and add a color/label to each specific section
    for(size_t t = 0; t < outputMat.rows; t++)
    {
        // The room: from frame 1 to 16 and from frame 372 to 392
        if ( t >= 0 && t < 16 || t >= 372 && t < 392 )
        {
            outputMat.row(t) += cv::Scalar(0x33,0,0xff); // red
        }
        
        // The outdoor corridor: from frame 16 to 63 and from frame 325 to 371
        if ( t >= 16 && t < 63 || t >= 325 && t < 371 )
        {
            outputMat.row(t) += cv::Scalar(0x33,0x66,0xff); // orange
        }
        
        // The indoor corridor: from frame 64 to 324
    }
    
    cv::addText(outputMat, 
                "The room", 
                cv::Point(outputMat.cols - 100,18),
                cv::fontQt("Arial", 12, cv::Scalar(0,0,0))
                );
    cv::addText(outputMat, 
                "The outdoor corridor", 
                cv::Point(outputMat.cols - 100,45),
                cv::fontQt("Arial", 12, cv::Scalar(0,0,0))
                );
}

double computeRotationDifference(const cv::Vec3f R1, const cv::Vec3f R2)
{
    // Compute the distance
    double 
        rotationDistance = 0.0;
    
    // Declare euler' form vectors
    cv::Matx13d
        rDiff(1, 3, CV_64F);
    rDiff(0) = rDiff(1) = rDiff(2) = 0.0;
    
    // Declare rotation maticies
    cv::Matx33d
        q1(3, 3, CV_64F), 
        q2(3, 3, CV_64F), 
        qDiff(3, 3, CV_64F);
    
    // Compute matricies
    cv::Rodrigues(cv::Vec3d(R1),q1);
    
    cv::Rodrigues(cv::Vec3d(R2),q2);
    
    // Compute q' = q1^(-1) * q2
    qDiff = q1.inv() * q2;
    
    // Return to vector representation
    cv::Mat 
        diff;
    cv::Rodrigues(qDiff, diff);
    
    rDiff(0) = diff.at<double>(0,0);
    rDiff(1) = diff.at<double>(0,1);
    rDiff(2) = diff.at<double>(0,2);
    
    // Compute rotation norm
    rotationDistance = cv::norm<double>(rDiff);
}

/** Displays the usage message
 */
int help(void)
{
    std::cout << "Usage: loopclosuredetector -s <settings.yml> -l <dataset.log>" << std::endl;
    return 0;
}