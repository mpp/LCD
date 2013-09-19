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
#include "MOSAIC/DescriptorsMatcher/descriptorsmatcher.h"

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
    
    LCD::VectorBoWMatcher
        vectorBM(fs);
    
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
    
    std::vector< std::vector<bool> >
        matchesVector;
    
    cv::Mat
        confusionMatrix;       
    
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
    
    
    double epsilon;
    fs["NNDR"]["epsilon"] >> epsilon;
    
    int minMatches;
    fs["loopDetectorOptions"]["minNumOfMatch"] >> minMatches;
        
    /////////////////////////////    
    /// Main loop
    
    // Loop the log file
    std::cout << std::endl << "-------" << std::endl << "Looping: " << logFileName << std::endl << "-------" << std::endl;
    
    // Debug variables
    int 
        frameCount = 0,
        mapCount = 0;
    
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
        
        //     std::cout << type << " ";
        
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
                        
                        matches.clear();
                        kpts.clear();
                        
                        // 2 - Setup the matcher
                        MOSAIC::DescriptorsMatcher *dm_ = new MOSAIC::DescriptorsMatcher(fs, frameA, frameB);
                        
                        // 3 - Set the transformation matrix between poses A and B
                        MOSAIC::SingleCameraTriangulator *sct_ = new MOSAIC::SingleCameraTriangulator(fs);
                        
                        cv::Matx44d gAB_;
                        sct_->setg12(tA, tB, rA, rB, gAB_);
                        
                        /// computation
                        // 2 - Compute the matches
                        cv::Mat
                            desc1, desc2;
                        std::vector<cv::KeyPoint>
                            kptsA_, kptsB_;
                        dm_->compareWithNNDR(epsilon, matches, kptsA_, kptsB_, desc1, desc2);

                        // 3 - Compute the triangulation
                        std::vector<bool>
                            outliersMask;   // Mask to distinguish between inliers (1) and outliers (0) match points. 
                        // As example those with negative z are outliers.
                        std::vector<cv::Vec3d>
                            triangulated_points_;
                        sct_->setKeypoints(kptsA_, kptsB_, matches);
                        sct_->triangulate(triangulated_points_, outliersMask); // 3D points are all inliers! The mask is for the matches
                        
                        if (triangulated_points_.size() < minMatches)
                        {
                            std::cout << "MOSAIC said: bad couple of images" << std::endl;
                            frameAtaken = false;
                            continue;
                        }
                        
                        // 4 - Visualizzo/salvo i match
                        cv::Mat
                            window;
                        std::vector<cv::Scalar> colors_;
                        MOSAIC::drawMatches(frameA, frameB, window, kptsA_, kptsB_, matches, colors_, outliersMask);
                        cv::imwrite("./matches/" + referenceFrameName + "_Matches.pgm", window);
                        
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