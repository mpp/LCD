/**
 * \file bowmatcher.h
 * \Author: Michele Marostica
 * \brief: This class will encapsulate a Flann based matcher to match a new 
 *         frame againist the map's frames
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

#ifndef BOWMATCHER_H
#define BOWMATCHER_H

#define OPENCV2P4

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

#ifdef OPENCV2P4
#include <opencv2/nonfree/nonfree.hpp>
#endif

namespace LCD{
    
    template<class T> class BoWMatcher
    {
    public:
//         BoWMatcher(cv::FileStorage &settings);
        
        /** Compare the frame to the map
         * @param [in] frame the frame to be compared to the map
         * @param [out] matches the comparison results
         * @param [out] bow the BoW of the frame
         * @param [out] kpts the key points of the frame
         */
        virtual void compare(cv::Mat &frame, std::vector< T > &matches, cv::Mat &bow, std::vector<cv::KeyPoint> &kpts, std::vector< std::vector < int > > &pointIDXOfCLusters, cv::Mat *completeDescriptors) = 0;
        
        /** Add a frame to the map, use it after compare(...)
         * @param [in] bow the bow to be added to the map
         */
        virtual void add(cv::Mat &bow) = 0;
    protected:
        
        /** generates a feature detector based on options in the settings file
         * @param [in] fs the YAML file with the settings
         */
        void generateDetector(cv::FileStorage &fs);
        
        /** generates a feature detector based on options in the settings file
         * @param [in] fs the YAML file with the settings
         */
        void generateExtractor(cv::FileStorage &fs);
    
    protected:
        
        cv::Ptr<cv::FeatureDetector>
            feature_detector_;              //> a feature detector
        
        cv::Ptr<cv::DescriptorExtractor>
            descriptor_extractor_;          //> a descriptor extractor
        
        cv::Mat
            vocabulary_,                    //> the vocabulary
            cl_tree_;                       //> the Chow-Liu tree
            
        cv::Ptr<cv::DescriptorMatcher>  
            matcher_;                       //> the descriptor matcher that is used by the BOWImgDescriptorExtractor
            
        cv::Ptr<cv::BOWImgDescriptorExtractor> 
            bide_;                          //> the object that extract the BOW descriptor from a frame
            
    };
    
    
    template<class T> 
    void BoWMatcher<T>::generateDetector(cv::FileStorage &fs) 
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
    
    template<class T> 
    void BoWMatcher<T>::generateExtractor(cv::FileStorage &fs)
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
        else 
        {
            std::cerr << "Could not create Descriptor Extractor. Please specify extractor type in settings file" << std::endl;
        }
        
    }
    
}   // namespace LCD

#endif // BOWMATCHER_H
