/**
 * \file flannbowmatcher
 * \Author: Michele Marostica
 * \brief: This class will compare the bow with the map using a FLANN based Matcher
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

#ifndef FLANNBOWMATCHER_H
#define FLANNBOWMATCHER_H

#include "bowmatcher.h"

namespace LCD {
    
    class FLANNBoWMatcher : public BoWMatcher< std::vector< cv::DMatch > >
    {
    public:
        FLANNBoWMatcher(cv::FileStorage &settings);
        
        /** Compare the frame to the map
         * @param [in] frame the frame to be compared to the map
         * @param [out] matches the comparison results
         * @param [out] bow the BoW of the frame
         * @param [out] kpts the key points of the frame
         */
        void compare(cv::Mat &frame, std::vector< std::vector<cv::DMatch> > &matches, cv::Mat &bow, std::vector<cv::KeyPoint> &kpts, std::vector< std::vector < int > > &pointIDXOfCLusters, cv::Mat *completeDescriptors);
        
        /** Add a frame to the map, use it after compare(...)
         * @param [in] bow the bow to be added to the map
         */
        void add(cv::Mat &bow);
        
    private:
        
        cv::FlannBasedMatcher
            bowMatcher_;                     //> the BoW matcher, it compares a new BoW to the map
    };
    
}

#endif // FLANNBOWMATCHER_H
