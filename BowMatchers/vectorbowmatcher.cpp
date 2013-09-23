/**
 * \file vectorbowmatcher.cpp
 * \Author: Michele Marostica
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

#include "vectorbowmatcher.h"

namespace LCD {
    
    VectorBoWMatcher::VectorBoWMatcher(const cv::FileStorage& settings) : 
            BoWMatcher< cv::DMatch >()
    {
//         cv::FileStorage fs;
//         std::string vocabPath = settings["FilePaths"]["Vocabulary"];
//         
//         //load the vocabulary
//         std::cout << "Loading Vocabulary" << std::endl;
//         
//         fs.open(vocabPath, cv::FileStorage::READ);
//         
//         fs["Vocabulary"] >> vocabulary_;
//         
//         if (vocabulary_.empty())
//         {
//             std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
//             exit(-1006);
//         }
//         fs.release();
    }
    
    void VectorBoWMatcher::compare(const cv::Mat& bow, std::vector< cv::DMatch >& matches) const
    {
        matches.clear();
        
        // Iterate the list and populate the matches vector
        int i = 0;
        for (std::vector<cv::Mat>::const_iterator l = bow_vector_.begin(); l != bow_vector_.end(); l++)
        {
            // Compute the distance;
            double distance = cv::norm(bow, (*l));
            cv::DMatch temp;
            temp.distance = distance;
            temp.imgIdx = i++;
            temp.queryIdx = -1;
            temp.trainIdx = -1;
            matches.push_back(temp);
        }
    }

    void VectorBoWMatcher::add(const cv::Mat& bow)
    {
        bow_vector_.push_back(bow);
    }


}
