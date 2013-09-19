/**
 * \file openfabmap.h
 * \Author: Michele Marostica
 * \brief: This class will setup an OpenFabMap (cv::of2) object and wrap its behaviour
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

#ifndef OPENFABMAP_H
#define OPENFABMAP_H

#define OPENCV2P4

#include <opencv2/opencv.hpp>

#include <boost/lexical_cast.hpp>

#ifdef OPENCV2P4
#include <opencv2/nonfree/nonfree.hpp>
#endif

#include "bowmatcher.h"

namespace LCD {
    
    class OpenFABMap : BoWMatcher<cv::of2::IMatch>
    {
    public:
        OpenFABMap(cv::FileStorage &settings);
        
        /** Call the compare method of OpenFABMap
         * @param [in] frame the frame to be compared to the map
         * @param [out] matches the comparison results
         * @param [out] bow the BoW of the frame
         * @param [out] kpts the key points of the frame
         */
        void compare(cv::Mat &frame, std::vector<cv::of2::IMatch> &matches, cv::Mat &bow, std::vector<cv::KeyPoint> &kpts, std::vector< std::vector < int > > &pointIDXOfCLusters, cv::Mat *completeDescriptors);
        
        /** Call the add method of OpenFABMap, use it after compare(...)
         * @param [in] bow the bow to be added to the map
         */
        void add(cv::Mat &bow);
        
    private:
        
//         /** generates a feature detector based on options in the settings file
//          * @param [in] fs the YAML file with the settings
//          */
//         void generateDetector(cv::FileStorage &fs);
//         
//         /** generates a feature detector based on options in the settings file
//          * @param [in] fs the YAML file with the settings
//          */
//         void generateExtractor(cv::FileStorage &fs);
        
    private:
        cv::of2::FabMap
            *open_fab_map_;                 //> an open FAB Map object
            
//         cv::Ptr<cv::FeatureDetector>
//             feature_detector_;              //> a feature detector
//         
//         cv::Ptr<cv::DescriptorExtractor>
//             descriptor_extractor_;          //> a descriptor extractor
//             
//         cv::Mat
//             vocabulary_,                    //> the vocabulary
//             cl_tree_;
    };

} // namespace LCD

#endif // OPENFABMAP_H
