/**
 * \file <filename>
 * \Author: Michele Marostica
 * \brief: <brief>
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

#ifndef BOWEXTRACTOR_H
#define BOWEXTRACTOR_H

#include <opencv2/opencv.hpp>

#include "../MOSAIC/mosaic.h"

class BOWExtractor
{
public:
    BOWExtractor(const cv::Ptr<cv::DescriptorMatcher>& dmatcher, const cv::FileStorage fs);
    void setVocabulary( const cv::Mat& vocabulary );
    
    /** Compute the BOW and the array of cluster's indices
     * @param [in] descriptors a matrix of descriptors, one in each row
     * @param [out] bow the computed BOW
     * @param [out] pointIdxsOfClusters the cluster's indices
     */
    void compute(const cv::Mat& descriptors, cv::Mat& bow, std::vector< std::vector< int > >& pointIdxsOfClusters);
    
//     void compute(const MOSAIC::framePosePackage &fpp, cv::Mat &bow,
//                  std::vector< cv::KeyPoint >& kpts,
//                  std::vector< std::vector< int > > &pointIdxsOfClusters, cv::Mat& descriptors);
private:
    cv::Mat vocabulary_;
    cv::Ptr<cv::DescriptorMatcher> dmatcher_;
    MOSAIC::MOSAIC *mo;
};

#endif // BOWEXTRACTOR_H
