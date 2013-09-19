#include <fstream>
#include <locale>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

#include "frameposelist.h"
#include "matchviewer.h"
#include "BowMatchers/openfabmap.h"

int help(void);
void groundTruthOverlay(const cv::Mat &inputMat, cv::Mat &outputMat);

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
    /// Setup OpenFabMap
    
    LCD::OpenFABMap 
        oFabMap(fs);
    
    cv::Mat
        actualFrame,
        similarFrame,
        bow;
        
    std::vector<cv::KeyPoint>
        kpts;
        
    cv::Mat 
        completeDescriptors;

    std::vector<cv::of2::IMatch>
        matches;
    
    std::vector<cv::of2::IMatch>::iterator
        l;
    
    std::vector< std::vector<bool> >
        matchesVector;
    
    cv::Mat
        confusionMatrix;   
        
    int 
        similarFrameMapIndex = -1;
        
    double
        maxMatch = 0.0;
        
    std::vector< std::vector < int > > 
        pointIDXOfCLusters;
        
    float 
        minimumLoopClosureValue = fs["loopDetectorOptions"]["minimumLoopClosureValue"];
    
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
        corrispondenceThreshold = 25,
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
//                 }
//                 
//                 if (frameCount % 100 == 0)
//                 {
                    actualFrame = cv::imread(basePath + frameName, CV_LOAD_IMAGE_GRAYSCALE);
                    cv::cvtColor(actualFrame, actualFrameBGR, CV_GRAY2BGR);
                    
                    /// Pass it to Open FABMap
                    matches.clear();
                    kpts.clear();
                    pointIDXOfCLusters.clear();

                    oFabMap.compare(actualFrame, matches, bow, kpts, pointIDXOfCLusters, &completeDescriptors);
                    
                    /// Get the most similar frame map index
                    bool new_place_max = true;
                    maxMatch = 0.0;
                    similarFrameMapIndex = -1;
                    for(l = matches.begin(); l != matches.end(); l++) 
                    {
                        if (l->match > maxMatch)
                        {
                            maxMatch = l->match;
                            similarFrameMapIndex = l->imgIdx;
                        }
                        
                        //test for new location maximum
                        if(l->match > matches.front().match) 
                        {
                            new_place_max = false;
                        }
                    }
                    
                    // Save the matches for the generation of the confusion matrix
                    std::vector<bool> binaryMatches;
                    //                     std::cout << "Binary matches: ";
                    for (size_t t = 0; t < matches.size(); t++)
                    {
                        // If new place put true in the pseudodiagonal
                        if (new_place_max)
                        {
                            if (t == matches.size() - 1)
                            {
                                binaryMatches.push_back(true);
                                //                                 std::cout << "256 - ";
                            }
                            else
                            {
                                binaryMatches.push_back(false);
                                //                                 std::cout << "0 - ";
                            }
                        }
                        // Else put true at the match index
                        else
                        {
                            if (t == similarFrameMapIndex)
                            {
                                binaryMatches.push_back(true);
                                //                                 std::cout << "256 - ";
                            }
                            else
                            {
                                binaryMatches.push_back(false);
                                //                                 std::cout << "0 - ";
                            }
                        }
                    }
                    //                     std::cout << std::endl;
                    matchesVector.push_back(binaryMatches);
                    
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
                    
                    matchviewer.update(window, similarFrameMapIndex, isMatchAccept, list, matches, leftInfoDisplay, rightInfoDisplay);
//                     cv::imshow("Window", window);
//                     cv::waitKey(15);
                    
                    cv::imwrite("/home/mpp/WorkspaceTesi/loop_dataset/Screenshots/LoopClosureScreenOF2_NNDR0.8/screen_" + boost::lexical_cast<std::string>(frameCount) + ".jpg",window);
                    std::cout << "Frame #:" << frameCount << std::endl;
                    
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
    
    // Draw the confusion matrix
    int rows = matchesVector.size();
    //     std::cout << "Dimensione vettore: " << rows << std::endl;
    confusionMatrix = cv::Mat(rows, rows, CV_16UC1, cv::Scalar(0));
    int x = 0, y = 0;
    for(std::vector< std::vector< bool > >::iterator j = matchesVector.begin(); j != matchesVector.end(); j++)
    {
        for(std::vector<bool>::iterator l = (*j).begin(); l != (*j).end(); l++) 
        {
            if ((*l))
            {
                confusionMatrix.at<short>(y,x) = std::numeric_limits< short >::max();
            }
            x++;
        }
        y++;
        x = 0;
    }
    
    //     std::cout << confusionMatrix << std::endl;
    
    // Apply the ground truth overlay
    cv::Mat 
    coloredConfusionMatrix(cv::Size(confusionMatrix.cols, confusionMatrix.rows), CV_16UC3, cv::Scalar(0,0,0));
    groundTruthOverlay(confusionMatrix, coloredConfusionMatrix);
    
    
    cv::namedWindow("Confusion Matrix");
    //     cv::resize(confusionMatrix, confusionMatrix, cv::Size(800,800));
    cv::imshow("Confusion Matrix", coloredConfusionMatrix);
    cv::waitKey();
    
    cv::imwrite("result_OpenFABMap2_NNDR0.8.jpg", coloredConfusionMatrix);
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


/** Displays the usage message
 */
int help(void)
{
    std::cout << "Usage: loopclosuredetector -s <settings.yml> -l <dataset.log>" << std::endl;
    return 0;
}