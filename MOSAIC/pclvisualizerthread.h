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

#ifndef PCLVISUALIZERTHREAD_H
#define PCLVISUALIZERTHREAD_H

#include <boost/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>

namespace MOSAIC {

class pclVisualizerThread
{
public:
    
    pclVisualizerThread();
    
    void operator()();
    
    void updateClouds(const std::vector< cv::Vec3d >& pointsGroup, const cv::Vec3d& normal, const cv::Scalar &color);
    
    void keepLastCloud();
    
//     void close();
    
private:
    bool
        *update_;
    
    boost::shared_ptr<boost::mutex>
        updateModelMutex_;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        cloud_,
        all_estimated_cloud_;
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>::Ptr 
        rgb_,
        all_estimated_rgb_;
    
    pcl::PointCloud<pcl::Normal>::Ptr
        normals_,
        all_estimated_normals_;
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
        viewer_;
};

} // namespace MOSAIC

#endif // PCLVISUALIZERTHREAD_H