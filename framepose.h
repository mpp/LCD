/**
 * \file framepose.h
 * \Author: Michele Marostica
 * \brief: This class is a structure that contains the information related to a frame
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

#ifndef FRAMEPOSE_H
#define FRAMEPOSE_H

#include <string>

#include <opencv2/opencv.hpp>
#include <boost/iterator/iterator_concepts.hpp>

namespace LCD {
    
    //TODO: see if it is better to separate Pose and Frame in 2 classes
    
    class FramePose
    {
        friend class FramePoseList;
        friend class MatchViewer;

    public:
        
        FramePose() {};
        FramePose(std::string framePath, cv::Vec3f &poseT, cv::Vec3f &poseR, unsigned int timestamp, bool isOnMap, int mapIndex);
        
        virtual ~FramePose();

        void print();
        
    private:
        
        unsigned int
            timestamp_;     //> The timestamp of the pose
        
        std::string
            frame_path_;    //> The absolute path + filename of the frame
        
        cv::Vec3f
            pose_T_,
            pose_R_;        //> The pose of the robot relative to the frame
        
        bool
            is_on_map_;     //> True if the frame has been added to the map
        
        int
            map_index_;     //> If the frame is on the map, this is the index
            
        cv::Mat
            bow_,           //> Bag of Word of the related frame
            complete_descriptors_;    //> a mat with the set of descriptors extracted from the frame       
        
        std::vector<cv::KeyPoint> 
            kpts_;          //> keypoints of the related frame
            
        std::vector< std::vector< int > >
            point_IDX_of_clusters_; //> point_IDX_of_clusters_[i] contains the index of keypoints relative to cluster i
    
        std::vector<cv::Vec3d>
            triangulated_points_;   //> 3D points of the features
    };

} // namespace LCD

#endif // FRAMEPOSE_H
