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

BOWExtractor::BOWExtractor(const cv::Ptr<cv::DescriptorMatcher>& dmatcher, const cv::FileStorage fs)
        : dmatcher_(dmatcher)
{
//     mo = new MOSAIC::MOSAIC(fs);
}

void BOWExtractor::setVocabulary(const cv::Mat& vocabulary)
{
    dmatcher_->clear();
    vocabulary_ = vocabulary;
    dmatcher_->add( std::vector<cv::Mat>(1, vocabulary) );
}

void BOWExtractor::compute(const cv::Mat& descriptors, cv::Mat& bow, std::vector< std::vector< int > >& pointIdxsOfClusters)
{
    bow.release();
    
    int clusterCount = vocabulary_.rows;
    
    // Match keypoint descriptors to cluster center (to vocabulary)
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


// void BOWExtractor::compute(const MOSAIC::framePosePackage& fpp, cv::Mat& bow, std::vector< cv::KeyPoint > &kpts, std::vector< std::vector< int > >& pointIdxsOfClusters, cv::Mat& descriptors)
// {
//     bow.release();
//     
//     int clusterCount = vocabulary_.rows;
//     
//     // Compute descriptors for the image.
//     mo->setFramesAndPoses(fpp);
//     
//     cv::Mat descriptorsBOW;
//     mo->computeDescriptors(kpts, descriptorsBOW, descriptors);
//     
//     if (kpts.empty())
//     {
//         std::cout << "BOWExtractor said: Bad couple of images." << std::endl;
//         return;
//     }
//     
//     // Match keypoint descriptors to cluster center (to vocabulary)
//     std::vector<cv::DMatch> matches;
//     dmatcher_->match( descriptorsBOW, matches );
//     
//     // Compute image descriptor
//     pointIdxsOfClusters.clear();
//     pointIdxsOfClusters.resize(clusterCount);
//     
//     bow = cv::Mat( 1, clusterCount, CV_32FC1, cv::Scalar::all(0.0) );
//     float *dptr = (float*)bow.data;
//     for( size_t i = 0; i < matches.size(); i++ )
//     {
//         int queryIdx = matches[i].queryIdx;
//         int trainIdx = matches[i].trainIdx; // cluster index
//         CV_Assert( queryIdx == (int)i );
//         
//         dptr[trainIdx] = dptr[trainIdx] + 1.f;
//         pointIdxsOfClusters[trainIdx].push_back( queryIdx );
//     }
//     
//     // Normalize image descriptor.
//     bow /= descriptorsBOW.rows;
// }
