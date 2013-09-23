/**
 * \file bowextractor.cpp
 * \Author: Michele Marostica
 *  
 * Copyright (c) 2013, Michele Marostica (michelemaro AT gmail DOT com)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1 - Redistributions of source code must retain the above copyright notice, 
 * 	   this list of conditions and the following disclaimer.
 * 2 - Redistributions in binary form must reproduce the above copyright 
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
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
 * 
 */

#include "bowextractor.h"

namespace LCD {

BOWExtractor::BOWExtractor(const cv::FileStorage fs)
{
    dmatcher_ = new cv::FlannBasedMatcher();
    
    cv::FileStorage vocab_path;
    cv::Mat vocabulary;
    std::string vocabPath = fs["FilePaths"]["Vocabulary"];
    
    //load the vocabulary
    std::cout << "Loading Vocabulary" << std::endl;
    
    vocab_path.open(vocabPath, cv::FileStorage::READ);
    
    vocab_path["Vocabulary"] >> vocabulary;
    
    if (vocabulary.empty())
    {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        exit(-1006);
    }
    
    setVocabulary(vocabulary);
    
    vocab_path.release();
}

void BOWExtractor::setVocabulary(const cv::Mat& vocabulary)
{
    dmatcher_->clear();
    std::cout << "righe vocabolario" << vocabulary.rows << std::endl;
    dmatcher_->add( std::vector<cv::Mat>(1, vocabulary) );
}

void BOWExtractor::compute(const cv::Mat& descriptors, cv::Mat& bow, std::vector< std::vector< int > >& pointIdxsOfClusters)
{
    bow.release();
    
    int clusterCount = dmatcher_->getTrainDescriptors().at(0).rows;
//     std::cout << "clustercount" << clusterCount << std::endl;
    
    // Match keypoint descriptors to cluster center (to vocabuslary)
    std::vector<cv::DMatch> matches;
    dmatcher_->match( descriptors, matches );
    
    // Compute image descriptor
    pointIdxsOfClusters.clear();
    pointIdxsOfClusters.resize(clusterCount);
    
    bow = cv::Mat( 1, clusterCount, CV_32FC1, cv::Scalar::all(0.0) );
    float *dptr = (float*)bow.data;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );
        
        dptr[trainIdx] = dptr[trainIdx] + 1.f;
        pointIdxsOfClusters[trainIdx].push_back( queryIdx );
    }
    
    // Normalize image descriptor.
    bow /= descriptors.rows;
}

} // namespace LCD
