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
    
    // BoWMatcher base class
    template<class T> class BoWMatcher
    {
    public:
        
        /** Compare the bow to the map
         * @param [in] bow the BoW of the frame
         * @param [out] matches the comparison results
         */
        virtual void compare(const cv::Mat &bow, std::vector< T > &matches) const = 0;
        
        /** Add a frame to the map, use it after compare(...)
         * @param [in] bow the bow to be added to the map
         */
        virtual void add(const cv::Mat &bow) = 0;
    
    protected:
        
//         cv::Mat
//             vocabulary_;                    //> the vocabulary
    };
    
    
}   // namespace LCD

#endif // BOWMATCHER_H
