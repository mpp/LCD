/**
 * \file vectorbowmatcher.cpp
 * \Author: Michele Marostica
 *  
 * Copyright (c) 2012, Michele Marostica (michelemaro@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * 1 - Redistributions of source code must retain the above copyright notice, 
 * 	   this list of conditions and the following disclaimer.
 * 2 - Redistributions in binary form must reproduce the above copyright 
 *	   notice, this list of conditions and the following disclaimer in the
 *	   documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "vectorbowmatcher.h"

namespace LCD {
    
    VectorBoWMatcher::VectorBoWMatcher(cv::FileStorage &settings) : 
            BoWMatcher< cv::DMatch >()
    {
        
        // Create detector and extractor
        generateDetector(settings);
        
        if(!feature_detector_) 
        {
            std::cerr << "Feature Detector error" << std::endl;
            exit(-1002);
        }
        
        generateExtractor(settings);
        
        if(!descriptor_extractor_) 
        {
            std::cerr << "Feature Extractor error" << std::endl;
            exit(-1003);
        }
        
        cv::FileStorage 
            fs;
        
        //load FabMap training data
        std::string 
            fabmapTrainDataPath = settings["FilePaths"]["TrainImagDesc"],
            chowliutreePath = settings["FilePaths"]["ChowLiuTree"],
            vocabPath = settings["FilePaths"]["Vocabulary"];
        
        std::cout << "Loading FabMap Training Data" << std::endl;
        
        fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
        
        cv::Mat 
            fabmapTrainData;
        
        fs["BOWImageDescs"] >> fabmapTrainData;
        
        if (fabmapTrainData.empty()) 
        {
            std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found" << std::endl;
            exit(-1004);
        }
        fs.release();
        
        //load a chow-liu tree
        std::cout << "Loading Chow-Liu Tree" << std::endl;
        
        fs.open(chowliutreePath, cv::FileStorage::READ);
        
        fs["ChowLiuTree"] >> cl_tree_;
        
        if (cl_tree_.empty())
        {
            std::cerr << chowliutreePath << ": Chow-Liu tree not found" << std::endl;
            exit(-1005);
        }
        fs.release();
        
        //load the vocabulary
        std::cout << "Loading Vocabulary" << std::endl;
        
        fs.open(vocabPath, cv::FileStorage::READ);
        
        fs["Vocabulary"] >> vocabulary_;
        
        if (vocabulary_.empty())
        {
            std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
            exit(-1006);
        }
        fs.release();
        
        matcher_ = cv::DescriptorMatcher::create("FlannBased");
        
        be_ = new BOWExtractor(matcher_, settings);
        be_->setVocabulary(vocabulary_);
    }
    
    void VectorBoWMatcher::compare(cv::Mat& frame, std::vector< cv::DMatch >& matches, cv::Mat& bow, std::vector< cv::KeyPoint >& kpts, std::vector< std::vector< int > >& pointIDXOfCLusters, cv::Mat* completeDescriptors)
    {
    }
//         // Extract the keypoints and the BoW
//         feature_detector_->detect(frame, kpts);
//         
//         bide_->compute(frame, kpts, bow, &pointIDXOfCLusters, completeDescriptors);
//         
//         
//         /// TODO: find a way to use the returned descriptors from the bide extractor
//         ///       why? because for now it returns a 0x0 matrix, so find out why :(
//         cv::SurfDescriptorExtractor 
//             extractor;
//         
//         extractor.compute(frame, kpts, *completeDescriptors);
//         
//         // Iterate the list and populate the matches vector
//         int i = 0;
//         for (std::vector<cv::Mat>::iterator l = bow_vector_.begin(); l != bow_vector_.end(); l++)
//         {
//             // Compute the distance;
//             double distance = cv::norm(bow, (*l));
//             cv::DMatch temp;
//             temp.distance = distance;
//             temp.imgIdx = i++;
//             temp.queryIdx = -1;
//             temp.trainIdx = -1;
//             matches.push_back(temp);
//         }
//         
//     }
    
    void VectorBoWMatcher::compare(cv::Mat& bow, std::vector< cv::DMatch >& matches)
    {
        matches.clear();
        
        // Iterate the list and populate the matches vector
        int i = 0;
        for (std::vector<cv::Mat>::iterator l = bow_vector_.begin(); l != bow_vector_.end(); l++)
        {
            // Compute the distance;
            double distance = cv::norm(bow, (*l));
            cv::DMatch temp;
            temp.distance = distance;
            temp.imgIdx = i++;
            temp.queryIdx = -1;
            temp.trainIdx = -1;
            matches.push_back(temp);
        }
    }
    
    void VectorBoWMatcher::computeBOW(const cv::Mat& descriptors, cv::Mat& bow, std::vector< std::vector< int > >& pointIdxsOfClusters)
    {
        be_->compute(descriptors, bow, pointIdxsOfClusters);
    }


    
//     void VectorBoWMatcher::compare(MOSAIC::framePosePackage &fpp, std::vector< cv::DMatch >& matches, cv::Mat& bow, std::vector< cv::KeyPoint >& kpts, std::vector< std::vector< int > >& pointIDXOfCLusters, cv::Mat &completeDescriptors)
//     {
//         be_->compute(fpp, bow, kpts, pointIDXOfCLusters, completeDescriptors);
//         
//         if (kpts.empty())
//         {
//             std::cout << "VectorBoWMatcher said: Bad couple of images" << std::endl;
//             return;
//         }
//         
//         // Iterate the list and populate the matches vector
//         int i = 0;
//         for (std::vector<cv::Mat>::iterator l = bow_vector_.begin(); l != bow_vector_.end(); l++)
//         {
//             // Compute the distance;
//             double distance = cv::norm(bow, (*l));
//             cv::DMatch temp;
//             temp.distance = distance;
//             temp.imgIdx = i++;
//             temp.queryIdx = -1;
//             temp.trainIdx = -1;
//             matches.push_back(temp);
//         }
//         
//     }


    void VectorBoWMatcher::add(cv::Mat& bow)
    {
        bow_vector_.push_back(bow);
    }


}
