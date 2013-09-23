/**
 * \file vectorbowmatcher.h
 * \Author: Michele Marostica
 * \brief: This class will compare the bow with the map using a KDTree
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

#ifndef VECTORBOWMATCHER_H
#define VECTORBOWMATCHER_H

#include "bowmatcher.h"
#include "bowextractor.h"

#include "../MOSAIC/mosaic.h"

namespace LCD {
    
    class VectorBoWMatcher : public BoWMatcher<cv::DMatch>
    {
    public:
        
        VectorBoWMatcher(const cv::FileStorage &settings);
        
        /** Compare the bow to the map
         * @param [in] bow the BoW of the frame
         * @param [out] matches the comparison results
         */
        void compare(const cv::Mat &bow, std::vector< cv::DMatch > &matches) const;
        
        /** Add a frame to the map, use it after compare(...)
         * @param [in] bow the bow to be added to the map
         */
        void add(const cv::Mat &bow);
    
    private:
    
        std::vector<cv::Mat>
            bow_vector_;
    };

} // namespace LCD

#endif // VECTORBOWMATCHER_H
