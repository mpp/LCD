#include <fstream>
#include <locale>
#include <iomanip>
#include <limits>

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

#include "frameposelist.h"
#include "matchviewer.h"
#include "BowMatchers/flannbowmatcher.h"

int help(void);

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
    
    LCD::FLANNBoWMatcher 
        flannBM(fs);
    
    cv::Mat
        actualFrame,
        similarFrame,
        bow;
    
    std::vector<cv::KeyPoint>
        kpts;
    
    cv::Mat 
        completeDescriptors;
    
    std::vector< std::vector<cv::DMatch> >
        matches;
    
    std::vector<cv::DMatch>::iterator
        l;
    
    std::vector< std::vector<cv::DMatch> >
        matchesVector;
    
    cv::Mat
        confusionMatrix;       
    
    int 
        similarFrameMapIndex = -1;
    
    double
        minDistance = std::numeric_limits< double >::max();
    
    std::vector< std::vector < int > > 
        pointIDXOfCLusters;
    
    /////////////////////////////    
    /// Setup the FramePoseList
    
    LCD::FramePoseList 
        list;
    
    std::string 
        temp,
        type,
        frameName;
    
    unsigned int
        timestamp;
    
    cv::Vec3f
        poseTranslation,
        poseRotation;
    
    bool 
        associateFramePos = false;
    
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
        corrispondenceThreshold = 25,   /// TODO: put this values in the configuration file
        corrispondenceCounter = 0;
    
    bool
        corrispondenceThresholdConstraint = false;
    
    /////////////////////////////    
    /// Main loop
    
    // Loop the log file
    std::cout << "Looping: " << logFileName << std::endl;
    
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
                //                 std::cout << poseTranslation(0) << " " << poseTranslation(1) << " " << poseTranslation(2) << " " << poseRotation(0) << " " << poseRotation(1) << " " << poseRotation(2) << std::endl;
                
                double
                    distance = list.distanceFromLastMapFrame(poseTranslation, poseRotation);
                //                 std::cout << "Frame #: " << frameCount-1 << "Distance from last map frame: " << distance << std::endl;
                
                if ( distance > distanceThreshold )
                {
                    actualFrame = cv::imread(basePath + frameName, CV_LOAD_IMAGE_GRAYSCALE);
                    cv::cvtColor(actualFrame, actualFrameBGR, CV_GRAY2BGR);
                    
                    /// Pass it to the FLANN BoW Matcher
                    matches.clear();
                    kpts.clear();
                    pointIDXOfCLusters.clear();
                    
                    std::cout << "DB1" << std::endl;
                    flannBM.compare(actualFrame, matches, bow, kpts, pointIDXOfCLusters, &completeDescriptors);
                    std::cout << "DB2" << std::endl;
                    flannBM.add(bow);
                    std::cout << "DB2" << std::endl;
                    
                    /// Get the most similar frame map index
                    bool new_place_max = true;
                    minDistance = std::numeric_limits< double >::max();
                    similarFrameMapIndex = -1;
                    
                    std::cout << "Distancies: ";
                    for(std::vector< std::vector< cv::DMatch > >::iterator j = matches.begin(); j != matches.end(); j++)
                    {
                        // Store in a vector to sucessively create the confusion matrix
                        matchesVector.push_back((*j));
                        
                        for(l = (*j).begin(); l != (*j).end(); l++) 
                        {
                            std::cout << l->distance << " - ";
                            if (l->distance < minDistance)
                            {
                                minDistance = l->distance;
                                similarFrameMapIndex = l->imgIdx;
                            }
                        }
                    }
                    std::cout << std::endl;
/*                    
                    std::string corrispondenceStr = "";
                    if (similarFrameMapIndex >= 0)
                    {
                        corrispondenceCounter = list.descriptorMatcher(&completeDescriptors, similarFrameMapIndex);
                        
                        corrispondenceStr = "# of similar descriptors: " 
                        + boost::lexical_cast<std::string>(corrispondenceCounter)
                        + "/" +
                        boost::lexical_cast<std::string>(kpts.size());
                    }
                    
                    std::string
                    rightInfoDisplay = "Most similar map frame: " 
                    + boost::lexical_cast<std::string>(similarFrameMapIndex)
                    + " - Match value: "
                    + boost::lexical_cast<std::string>(maxMatch),
                    leftInfoDisplay =  "Frame # " 
                    + boost::lexical_cast<std::string>(frameCount-1)
                    + " - "
                    + corrispondenceStr;
                    
                    /// Add to the list and to openfabmap
                    bool isMatchAccept = false;
                    if(new_place_max || maxMatch < minimumLoopClosureValue || corrispondenceCounter < corrispondenceThreshold) 
                    {
                        oFabMap.add(bow);
                        list.add(basePath + frameName, poseTranslation, poseRotation, timestamp, true, mapCount++, &bow, &kpts, &pointIDXOfCLusters, &completeDescriptors);
                    }
                    else
                    {
                        list.add(basePath + frameName, poseTranslation, poseRotation, timestamp, false, -1, &bow, &kpts, &pointIDXOfCLusters, &completeDescriptors);
                        isMatchAccept = true;
                    }
                    #ifdef OFABMAP                                                
                    matchviewer.update(window, similarFrameMapIndex, isMatchAccept, list, matches, leftInfoDisplay, rightInfoDisplay);
                    #endif
                    //                     cv::imshow("Window", window);
                    //                     cv::waitKey(15);
                    
                    //                     cv::imwrite("/home/mpp/WorkspaceTesi/loop_dataset/LoopClosureScreenDM_NNDR0.65_FC/screen_" + boost::lexical_cast<std::string>(frameCount) + ".jpg",window);
                    std::cout << "Frame #:" << frameCount << std::endl;
                    */
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
    
//     // Draw the confusion matrix
//     int cols = (*(matchesVector.end() -1)).size();
//     int rows = matchesVector.size();
//     confusionMatrix = cv::Mat(cols, rows, CV_32FC1, cv::Scalar(0));
//     int x = 0, y = 0;
//     for(std::vector< std::vector< cv::DMatch > >::iterator j = matchesVector.begin(); j != matchesVector.end(); j++)
//     {
//         for(l = (*j).begin(); l != (*j).end(); l++) 
//         {
//             if (x = 0)
//             {
//                 confusionMatrix.at<float>((*j).size(),y) = l->distance;
//             }
//             confusionMatrix.at<float>((*l).imgIdx,y) = l->distance;
//             x++;
//         }
//         y++;
//         x = 0;
//     }
//     
//     std::cout << confusionMatrix << std::endl;
    
    //     cv::namedWindow("Confusion Matrix");
    //     confusionMatrix.resize();
    
    std::cout << "Looping end. See you." << std::endl;  
    
    
    return 0;
}



/** Displays the usage message
 */
int help(void)
{
    std::cout << "Usage: loopclosuredetector -s <settings.yml> -l <dataset.log>" << std::endl;
    return 0;
}