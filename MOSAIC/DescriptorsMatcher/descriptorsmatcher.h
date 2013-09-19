/**
 * \file descriptorsmatcher.h
 * \Author: Michele Marostica
 * \brief: The aim is to compare two frame by extracting features and matching them with the NNDR metric.
 *         The objective is to collect the results of different descriptors with different parameters to evaluate their performaces.
 *  
 * Copyright (c) 2012, Michele Marostica (michelemaro@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * 1 - Redistributions of source code must retain the above copyright notice, 
 *     this list of conditions and the following disclaimer.
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
 */

#ifndef DESCRIPTORSMATCHER_H
#define DESCRIPTORSMATCHER_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp> // For SIFT and SURF

namespace MOSAIC {
    
class DescriptorsMatcher
{
public:
    
    /** The aim is to compare two frame
     * @param [in] fs the YAML file with the settings
     * @param [in] frame_a/b the frames to be compared
     */
    DescriptorsMatcher(const cv::FileStorage &fs, cv::Mat &frame_a, cv::Mat &frame_b);
    
    DescriptorsMatcher(const cv::FileStorage &fs);
    
    void setImages(cv::Mat &frame_a, cv::Mat &frame_b);
    
    /** Cross-compare two frame
     * @param [out] matchesAB the forward comparison results
     * @param [out] matchesBA the forward comparison results
     * @param [out] kpts_a/b the key points of the frame a or b
     * @param [out] completeDescriptors_a/b the descriptors of the frame a or b
     */
    void crosscompare( std::vector< std::vector< cv::DMatch > > &matchesAB, std::vector< std::vector< cv::DMatch > > &matchesBA, 
                       std::vector<cv::KeyPoint> &kpts_a, std::vector<cv::KeyPoint> &kpts_b, 
                       cv::Mat &completeDescriptors_a, cv::Mat &completeDescriptors_b);
    
    
    /** Compare two frame
     * @param [out] matches the comparison results (knnmatch)
     * @param [out] kpts_a/b the key points of the frame a or b
     * @param [out] completeDescriptors_a/b the descriptors of the frame a or b
     */
    void compare( std::vector< std::vector< cv::DMatch > > &matches, 
                  std::vector<cv::KeyPoint> &kpts_a, std::vector<cv::KeyPoint> &kpts_b, 
                  cv::Mat &completeDescriptors_a, cv::Mat &completeDescriptors_b);
    
    /** Compare two frame
     * @param [in] epsilon the NNDR threshold
     * @param [out] matches the comparison results after NNDR application
     * @param [out] kpts_a/b the key points of the frame a or b
     * @param [out] completeDescriptors_a/b the descriptors of the frame a or b
     */
    void compareWithNNDR( double epsilon, std::vector< cv::DMatch > &matches, 
                          std::vector<cv::KeyPoint> &kpts_a, std::vector<cv::KeyPoint> &kpts_b, 
                          cv::Mat &completeDescriptors_a, cv::Mat &completeDescriptors_b);
    
    void extractDescriptorsFromPatches(const std::vector<cv::Mat> &patchesVector, cv::Mat &descriptors);
    void extractDescriptors128FromPatches(const std::vector<cv::Mat> &patchesVector, cv::Mat &descriptors128);
    
protected:
    /** Generates a feature detector based on options in the settings file
     * @param [in] fs the YAML file with the settings
     */
    void generateDetector(const cv::FileStorage &fs);
    
    /** Generates a feature detector based on options in the settings file
     * @param [in] fs the YAML file with the settings
     */
    void generateExtractor(const cv::FileStorage &fs);
    
protected:
    
    
    cv::Ptr<cv::Mat>
        image_a_,                        //> the first image to be compared
        image_b_;                        //> the second image
    
    cv::Ptr<cv::FeatureDetector>
        feature_detector_;              //> a feature detector
    
    cv::Ptr<cv::DescriptorExtractor>
        descriptor_extractor_,          //> a descriptor extractor
        descriptor_extractor_128_;          //> a descriptor extractor -> SURF 128bit
    
    cv::Ptr<cv::DescriptorMatcher>   
        matcher_;                       //> the matcher class
    
    std::vector< std::vector< cv::DMatch > >
        matches_;                       //> a vector containing the results of a knn match
    
};

} // namespace MOSAIC

#endif // DESCRIPTORSMATCHER_H