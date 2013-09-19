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

typedef struct framePair_ 
{
    cv::Vec3f tA,tB,rA,rB;
    std::string fA,fB;
    cv::Mat bow, descriptors;
    std::vector< std::vector<int> > IDX;
    std::vector< cv::KeyPoint > kpts;
} framePair;

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
        descriptors;
    
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
        bestDistance = std::numeric_limits< double >::max();
    
    /////////////////////////////    
    /// Setup the FramePoseList
    
    LCD::FramePoseList 
        list(fs);
    
    std::string 
        frameName,
        referenceFrameName;
        
        
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
    int
        corrispondenceThreshold = 0, 
        corrispondenceCounter = 0;
    double
        maxBOWDistanceThreshold;
        
    fs["loopDetectorOptions"]["scoreThreshold"] >> corrispondenceThreshold;
    fs["loopDetectorOptions"]["maxBOWDistanceThreshold"] >> maxBOWDistanceThreshold;
        
    
    bool
        corrispondenceThresholdConstraint = false;
        
    /////////////////////////////
    /// Collect the precoputed descriptors
        
    std::string
        descriptorsFileName = "descriptors_list.txt";
    
    std::ifstream 
        descriptorsFile(descriptorsFileName.c_str());
    
    if (!descriptorsFile)
    {
        std::cerr << "Cannot open file" << descriptorsFileName << std::endl;
        exit(-1);
    }
    
    // iterate throught each file in the descriptors directory and load the data
    std::vector< framePair >
        precoputedValues;
        
    std::string
        file;
    while (!descriptorsFile.eof())
    {
        descriptorsFile >> file;
        
        cv::FileStorage
            descriptorsContainer("descriptors/" + file, cv::FileStorage::READ);
            
        framePair fp;
            
        descriptorsContainer["Descriptors"] >> fp.descriptors;
        descriptorsContainer["BOW"] >> fp.bow;
        descriptorsContainer["tA"][0] >> fp.tA[0];descriptorsContainer["tA"][1] >> fp.tA[1];descriptorsContainer["tA"][2] >> fp.tA[2];
        descriptorsContainer["tB"][0] >> fp.tB[0];descriptorsContainer["tB"][1] >> fp.tB[1];descriptorsContainer["tB"][2] >> fp.tB[2];
        descriptorsContainer["rA"][0] >> fp.rA[0];descriptorsContainer["rA"][1] >> fp.rA[1];descriptorsContainer["rA"][2] >> fp.rA[2];
        descriptorsContainer["rB"][0] >> fp.rB[0];descriptorsContainer["rB"][1] >> fp.rB[1];descriptorsContainer["rB"][2] >> fp.rB[2];
        descriptorsContainer["ReferenceFrame"] >> fp.fA;
        descriptorsContainer["SupportFrame"] >> fp.fB;
        
        cv::FileNode KPTSVector =descriptorsContainer["KPTS"];
        cv::read(KPTSVector, fp.kpts);
        
        cv::FileNode IDXVector = descriptorsContainer["IDX"];
        
        cv::FileNodeIterator it = IDXVector.begin();
        
        int idx = 0;
        fp.IDX = std::vector< std::vector<int> >();
        fp.IDX.resize(IDXVector.size());
        
        for (; it != IDXVector.end(); it++, idx++)
        {
            (*it) >> fp.IDX[idx];
        }
        
        precoputedValues.push_back(fp);
    }
        
    /////////////////////////////    
    /// Main loop
    
    // Loop the log file
    std::cout << std::endl << "-------" << std::endl << "Looping " << std::endl << "-------" << std::endl << std::endl;
    
    // Debug variables
    int 
        frameCount = 0,
        mapCount = 0;
        
    std::vector< framePair >::iterator
        descIT = precoputedValues.begin();
    
    while (descIT != precoputedValues.end())
    {
        /// Pass it to the FLANN BoW Matcher
        matches.clear();
        
        cv::Mat 
            completeDescriptors = descIT->descriptors.clone(),
            bow = descIT->bow.clone();
        
        if (bow.empty() || completeDescriptors.empty() || completeDescriptors.rows <= 1)
        {
            descIT++;
            continue;
        }
            
        vectorBM.compare(bow, matches); 
        
        // Sort the matches vector
        std::sort(matches.begin(), matches.end());
        
        // Consider the first k elements, of them compute the descriptor constraint and take the one with the best score
        /// Get the most similar frame map index
        bool newPlace = false;
        similarFrameMapIndex = -1;
        bestScore = -1;
        bestDistance = -1;
        
        int k = 4; /// TODO: move this value to the setting file
        
        std::vector< std::pair<int, int> > matchesIndices;
        
        std::cout << "Frame " << descIT->fA << " scores:"; 
        for (size_t i = 0; i < k; i++)
        {
            // Check boundaries
            if (i >= matches.size())
            {
                break;
            }
            
            std::vector< std::pair<int, int> > temp;
            
            int score = list.descriptorMatcher(&completeDescriptors, matches[i].imgIdx, temp);
            
            std::cout << " (" << matches[i].imgIdx << ", " << score << ", " << matches[i].distance << ") - ";
            
            if (score > bestScore /*&& matches[i].distance < maxBOWDistanceThreshold*/)
            {
                similarFrameMapIndex = matches[i].imgIdx;
                bestScore = score;
                bestDistance = matches[i].distance;
                
                matchesIndices = std::vector< std::pair<int,int> >(temp.begin(), temp.end());
            }
        }
        std::cout << std::endl;

        std::cout << "Best match: index " << similarFrameMapIndex << ", score " << bestScore << ", distance " << bestDistance << std::endl;
        
        cv::Vec3f
            fakeT(descIT->tA),
            fakeR(descIT->rA);
            
        frameName = descIT->fA;
            
        int fakeTimestamp = 0;
        
        if (similarFrameMapIndex < 0 || bestScore <= corrispondenceThreshold)
        {
            newPlace = true;
            vectorBM.add(bow);
            list.add(basePath + frameName, fakeT, fakeR, fakeTimestamp, true, mapCount++, &bow, &(descIT->kpts), &(descIT->IDX), &completeDescriptors);
        }
        else
        {
            cv::Mat accorpatedDescriptors;
//       list.addDescriptorsToMapFrame(similarFrameMapIndex, &(descIT->kpts), &(descIT->IDX), &completeDescriptors);
            list.addDescriptorsToMapFrame(similarFrameMapIndex, &(descIT->kpts), &(descIT->IDX), &completeDescriptors, &matchesIndices);
            list.add(basePath + frameName, fakeT, fakeR, fakeTimestamp, false, -1, &bow, &(descIT->kpts), &(descIT->IDX), &completeDescriptors);
            
//             /// REDO The match againist the map
//             std::cout << "Repeat the  " << descIT->fA << " scores:"; 
//             int updatedSimilarFrameMapIndex = 0;
//             for (size_t i = 0; i < k; i++)
//             {
//                 // Check boundaries
//                 if (i >= matches.size())
//                 {
//                     break;
//                 }
//                 
//                 // Avoid to check with the same frame in the map
//                 if (matches[i].imgIdx == similarFrameMapIndex)
//                 {
//                     continue;
//                 }
//                 
//                 std::vector< std::pair<int, int> > temp;
//                 
//                 int score = list.descriptorMatcher(&accorpatedDescriptors, matches[i].imgIdx, temp);
//                 
//                 std::cout << " (" << matches[i].imgIdx << ", " << score << ", " << matches[i].distance << ") - ";
//                 
//                 if (score > bestScore /*&& matches[i].distance < maxBOWDistanceThreshold*/)
//                 {
//                     updatedSimilarFrameMapIndex = matches[i].imgIdx;
//                     bestScore = score;
//                     bestDistance = matches[i].distance;
//                     
//                     matchesIndices = std::vector< std::pair<int,int> >(temp.begin(), temp.end());
//                 }
//             }
//             std::cout << std::endl;
// 
//             std::cout << "NEW Best match: index " << updatedSimilarFrameMapIndex << ", score " << bestScore << ", distance " << bestDistance << std::endl;
//         
//             // hack for following code:
//             similarFrameMapIndex = updatedSimilarFrameMapIndex;
        }
        
        
        // Save the matches for the generation of the confusion matrix
        std::vector<bool> binaryMatches;
        //                     std::cout << "Binary matches: ";
        for (size_t t = 0; t < matches.size(); t++)
        {
            // If new place put true in the pseudodiagonal
            if (newPlace)
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
        
//                     std::cout << "Map frames: " << mapCount << std::endl;
                            
        std::string corrispondenceStr =  "# of similar descriptors: " 
                                + boost::lexical_cast<std::string>(bestScore)
                                + "/" + boost::lexical_cast<std::string>(completeDescriptors.rows);
        
        std::string
            rightInfoDisplay = "Most similar map frame: " 
                                + boost::lexical_cast<std::string>(similarFrameMapIndex)
                                + " - with distance: "
                                + boost::lexical_cast<std::string>(bestDistance),
            leftInfoDisplay =  "Frame # " 
                                + boost::lexical_cast<std::string>(frameCount)
                                + " - "
                                + corrispondenceStr;
        
        matchviewer.update(window, similarFrameMapIndex, !newPlace, list, matches, matchesIndices, leftInfoDisplay, rightInfoDisplay);

        cv::imshow("Window", window);
        cv::waitKey(150);
        cv::imwrite("Screenshots/screen_" + boost::lexical_cast<std::string>(frameCount) + ".jpg",window);
        std::cout << "Frame #:" << frameCount << std::endl;
     
        frameCount++;
        descIT++;
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
    
    cv::imwrite("result_vector_MOSAIC.pgm", confusionMatrix);
    
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