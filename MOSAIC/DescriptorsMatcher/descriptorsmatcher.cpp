/**
 * \file descriptorsmatcher.cpp
 * \Author: Michele Marostica
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

#ifndef OPENCV2P4
#define OPENCV2P4
#endif

#include "descriptorsmatcher.h"

namespace MOSAIC {
    
DescriptorsMatcher::DescriptorsMatcher(const cv::FileStorage& fs, cv::Mat& frame_a, cv::Mat& frame_b)
{
    
    image_a_ = new cv::Mat(frame_a);
    image_b_ = new cv::Mat(frame_b);
    
    generateDetector(fs);
    
    if(!feature_detector_) 
    {
        std::cerr << "Feature Detector error" << std::endl;
        exit(-1002);
    }
    
    generateExtractor(fs);
    
    if(!descriptor_extractor_) 
    {
        std::cerr << "Feature Extractor error" << std::endl;
        exit(-1003);
    }
    
    std::string
        extractorType;
    
    fs["FeatureOptions"]["ExtractorType"] >> extractorType;
    
    if (extractorType == "ORB" || extractorType == "BRISK" || extractorType == "FREAK")
    {
        matcher_ = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 24, 2));
    }
    else
    {
        matcher_ = cv::DescriptorMatcher::create("FlannBased");
    }
}

DescriptorsMatcher::DescriptorsMatcher(const cv::FileStorage& fs)
{
    generateDetector(fs);
    
    if(!feature_detector_) 
    {
        std::cerr << "Feature Detector error" << std::endl;
        exit(-1002);
    }
    
    generateExtractor(fs);
    
    if(!descriptor_extractor_) 
    {
        std::cerr << "Feature Extractor error" << std::endl;
        exit(-1003);
    }
    
    std::string
    extractorType;
    
    fs["FeatureOptions"]["ExtractorType"] >> extractorType;
    
    if (extractorType == "ORB" || extractorType == "BRISK" || extractorType == "FREAK")
    {
        matcher_ = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(5, 24, 2));
    }
    else
    {
        matcher_ = cv::DescriptorMatcher::create("FlannBased");
    }
}

void DescriptorsMatcher::setImages(cv::Mat& frame_a, cv::Mat& frame_b)
{
    image_a_ = new cv::Mat(frame_a);
    image_b_ = new cv::Mat(frame_b);
}

void DescriptorsMatcher::crosscompare(std::vector< std::vector< cv::DMatch > >& matchesAB, std::vector< std::vector< cv::DMatch > >& matchesBA, std::vector< cv::KeyPoint >& kpts_a, std::vector< cv::KeyPoint >& kpts_b, cv::Mat& completeDescriptors_a, cv::Mat& completeDescriptors_b)
{
    // Detect keypoints
    feature_detector_->detect(*image_a_, kpts_a);
    feature_detector_->detect(*image_b_, kpts_b);
    
    // Extract feature descriptors from keypoints
    descriptor_extractor_->compute(*image_a_, kpts_a, completeDescriptors_a);
    descriptor_extractor_->compute(*image_b_, kpts_b, completeDescriptors_b);
    
    matcher_->knnMatch(completeDescriptors_a, completeDescriptors_b, matchesAB, 2);
    matcher_->knnMatch(completeDescriptors_b, completeDescriptors_a, matchesBA, 2);
    
}

void DescriptorsMatcher::compare(std::vector< std::vector< cv::DMatch > >& matches, 
                                 std::vector< cv::KeyPoint >& kpts_a, std::vector< cv::KeyPoint >& kpts_b, 
                                 cv::Mat &completeDescriptors_a, cv::Mat &completeDescriptors_b)
{
    // Detect keypoints
    feature_detector_->detect(*image_a_, kpts_a);
    feature_detector_->detect(*image_b_, kpts_b);
    
    // Extract feature descriptors from keypoints
    descriptor_extractor_->compute(*image_a_, kpts_a, completeDescriptors_a);
    descriptor_extractor_->compute(*image_b_, kpts_b, completeDescriptors_b);
    
    matcher_->knnMatch(completeDescriptors_a, completeDescriptors_b, matches_, 2);
    
    matches = matches_;
    
}

void DescriptorsMatcher::compareWithNNDR(double epsilon, std::vector< cv::DMatch >& matches, std::vector< cv::KeyPoint >& kpts_a, std::vector< cv::KeyPoint >& kpts_b, cv::Mat& completeDescriptors_a, cv::Mat& completeDescriptors_b)
{
    // Detect keypoints
    feature_detector_->detect(*image_a_, kpts_a);
    feature_detector_->detect(*image_b_, kpts_b);
    
//     for (std::size_t t = 0; t < kpts_a.size(); t++)
//     {
//         std::cout << "Keypoint: " << kpts_a[t].pt << " size: " << kpts_a[t].size << " octave: " << kpts_a[t].octave << std::endl;
//     }
    
    // Extract feature descriptors from keypoints
    descriptor_extractor_->compute(*image_a_, kpts_a, completeDescriptors_a);
    descriptor_extractor_->compute(*image_b_, kpts_b, completeDescriptors_b);
    
    matcher_->knnMatch(completeDescriptors_a, completeDescriptors_b, matches_, 2);
    
    for (std::vector< std::vector< cv::DMatch> >::iterator j = matches_.begin(); j != matches_.end(); j++)
    {
        if ((*j).size() >= 2)
        {
            if ((*j).at(0).distance <= epsilon * (*j).at(1).distance)
            {
                // Take the match
                matches.push_back((*j).at(0));
            }
        }
    }
    
}

void DescriptorsMatcher::extractDescriptorsFromPatches(const std::vector< cv::Mat >& patchesVector, cv::Mat& descriptors)
{
    std::vector<cv::Mat>::const_iterator patchIT = patchesVector.begin();
    
    std::vector<cv::Mat> descriptorsVector;
    
    std::vector< std::vector<cv::KeyPoint> > kpVector;
    
    while (patchIT != patchesVector.end())
    {
        int patchSize = patchIT->rows;
        int center = (int) floor(patchSize / 2);
        
        std::vector<cv::KeyPoint> kpts;
        
        // Setup the KeyPoint
        cv::KeyPoint kp;
        kp.pt = cv::Point2f(center, center);
        kp.size = patchSize;
        kp.angle = -1;
        kp.response = 1;
        kp.octave = 0;
        kp.class_id = 0;
        
        kpts.push_back(kp);
        kpVector.push_back(kpts);
        
        patchIT++;
    }
    
    // Compute the descriptor
    descriptor_extractor_->compute(patchesVector, kpVector, descriptorsVector);
    
    descriptors = cv::Mat::zeros(cv::Size(descriptorsVector[0].cols, patchesVector.size()), descriptorsVector[0].type());
    
    for (std::size_t r = 0; r < descriptors.rows; r++)
    {
        descriptorsVector[r].copyTo(descriptors.row(r));
        //         std::cout << descriptorsVector[r] << std::endl << descriptors.row(r) << std::endl << "----------------" << std::endl;
    }
    
}


void DescriptorsMatcher::extractDescriptors128FromPatches(const std::vector< cv::Mat >& patchesVector, cv::Mat& descriptors)
{
    std::vector<cv::Mat>::const_iterator patchIT = patchesVector.begin();
    
    std::vector<cv::Mat> descriptorsVector;
    
    std::vector< std::vector<cv::KeyPoint> > kpVector;
    
    while (patchIT != patchesVector.end())
    {
        int patchSize = patchIT->rows;
        int center = (int) floor(patchSize / 2);
        
        std::vector<cv::KeyPoint> kpts;
        
        // Setup the KeyPoint
        cv::KeyPoint kp;
        kp.pt = cv::Point2f(center, center);
        kp.size = patchSize;
        kp.angle = -1;
        kp.response = 1;
        kp.octave = 0;
        kp.class_id = 0;
        
        kpts.push_back(kp);
        kpVector.push_back(kpts);
        
        patchIT++;
    }
    
    descriptor_extractor_128_ = new cv::SURF( 400, 4, 2, true, true);
    
    // Compute the descriptor
    descriptor_extractor_128_->compute(patchesVector, kpVector, descriptorsVector);
    
    descriptors = cv::Mat::zeros(cv::Size(descriptorsVector[0].cols, patchesVector.size()), descriptorsVector[0].type());
    
    for (std::size_t r = 0; r < descriptors.rows; r++)
    {
        descriptorsVector[r].copyTo(descriptors.row(r));
        //         std::cout << descriptorsVector[r] << std::endl << descriptors.row(r) << std::endl << "----------------" << std::endl;
    }
    
}

void DescriptorsMatcher::generateDetector(const cv::FileStorage& fs) 
{   
    //create common feature detector and descriptor extractor
    std::string 
        detectorMode = fs["FeatureOptions"]["DetectorMode"],
        detectorType = fs["FeatureOptions"]["DetectorType"];
    
    feature_detector_ = NULL;
    
    if(detectorMode == "ADAPTIVE") 
    {
        
        if(detectorType != "STAR" && detectorType != "SURF" && detectorType != "FAST") 
        {
            std::cerr << "Adaptive Detectors only work with STAR, SURF "
            "and FAST" << std::endl;
        }
        else 
        {
            feature_detector_ = new cv::DynamicAdaptedFeatureDetector(cv::AdjusterAdapter::create(detectorType),
                                                                      fs["FeatureOptions"]["Adaptive"]["MinFeatures"], 
                                                                      fs["FeatureOptions"]["Adaptive"]["MaxFeatures"], 
                                                                      fs["FeatureOptions"]["Adaptive"]["MaxIters"]
            );
        }
    } 
    else if(detectorMode == "STATIC") 
    {
        if(detectorType == "STAR") 
        {
            
            feature_detector_ = new cv::StarFeatureDetector(
                fs["FeatureOptions"]["StarDetector"]["MaxSize"],
                fs["FeatureOptions"]["StarDetector"]["Response"],
                fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
                fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
                fs["FeatureOptions"]["StarDetector"]["Suppression"]);
            
        }
        else if(detectorType == "FAST")
        {
            
            feature_detector_ = new cv::FastFeatureDetector(
                fs["FeatureOptions"]["FastDetector"]["Threshold"],
                (int)fs["FeatureOptions"]["FastDetector"]
                ["NonMaxSuppression"] > 0);     
            
        } 
        else if(detectorType == "SURF") 
        {
            
            #ifdef OPENCV2P4
            feature_detector_ = new cv::SURF(
                fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                                             (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
            
            #else
            feature_detector_ = new cv::SurfFeatureDetector(
                fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
            #endif
        } 
        else if(detectorType == "SIFT") 
        {
            #ifdef OPENCV2P4
            feature_detector_ = new cv::SIFT(
                fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
            #else
            feature_detector_ = new cv::SiftFeatureDetector(
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"]);
            #endif
        }
        else if(detectorType == "MSER") 
        {
            
            feature_detector_ = new cv::MserFeatureDetector(
                fs["FeatureOptions"]["MSERDetector"]["Delta"],
                fs["FeatureOptions"]["MSERDetector"]["MinArea"],
                fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
                fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
                fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
                fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
                fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
                fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
                fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
            
        }
        else if(detectorType == "ORB") 
        {
            
            feature_detector_ = new cv::OrbFeatureDetector(
                fs["FeatureOptions"]["OrbDetector"]["NumFeatures"],
                fs["FeatureOptions"]["OrbDetector"]["ScaleFactor"],
                fs["FeatureOptions"]["OrbDetector"]["NumLevels"]);
            
        }
        else 
        {
            std::cerr << "Could not create detector class. Specify detector "
            "options in the settings file" << std::endl;
        }
    } 
    else 
    {
        std::cerr << "Could not create detector class. Specify detector "
        "mode (static/adaptive) in the settings file" << std::endl;
    }
}

void DescriptorsMatcher::generateExtractor(const cv::FileStorage& fs)
{
    std::string 
        extractorType = fs["FeatureOptions"]["ExtractorType"];
    
    descriptor_extractor_ = NULL;
    
    if(extractorType == "SIFT") 
    {
        #ifdef OPENCV2P4
        descriptor_extractor_ = new cv::SIFT(
            fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
            fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
            fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
            fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
            fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
        #else
        descriptor_extractor_ = new cv::SiftDescriptorExtractor();
        #endif
        
    } 
    else if(extractorType == "SURF") 
    {
        
        #ifdef OPENCV2P4
        descriptor_extractor_ = new cv::SURF(
            fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
            fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
            fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
            (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                                             (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
        
        #else
        descriptor_extractor_ = new cv::SurfDescriptorExtractor(
            fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
            fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
            (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                                                                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
        #endif
        
    } 
    else if(extractorType == "ORB") 
    {
        descriptor_extractor_ = new cv::ORB(
            fs["FeatureOptions"]["OrbDetector"]["NumFeatures"],
            fs["FeatureOptions"]["OrbDetector"]["ScaleFactor"],
            fs["FeatureOptions"]["OrbDetector"]["NumLevels"]);
    } 
    else if(extractorType == "BRISK") 
    {
        descriptor_extractor_ = new cv::BRISK(
            fs["FeatureOptions"]["BriskDetector"]["Threshold"],
            fs["FeatureOptions"]["BriskDetector"]["Octaves"]);
        
    } 
    else if(extractorType == "FREAK") 
    {
        descriptor_extractor_ = new cv::FREAK();
    } 
    else 
    {
        std::cerr << "Could not create Descriptor Extractor. Please specify extractor type in settings file" << std::endl;
    }
    
}

} //namespace MOSAIC