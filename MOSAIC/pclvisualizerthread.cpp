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

#include "pclvisualizerthread.h"

namespace MOSAIC {
    
pclVisualizerThread::pclVisualizerThread()
{
    update_ = new bool(false);
    
    updateModelMutex_ = boost::shared_ptr<boost::mutex>(new boost::mutex());
    
    viewer_ = boost::shared_ptr<pcl::visualization::PCLVisualizer>(
        new pcl::visualization::PCLVisualizer("Normal optimization viewer"));
    
    cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    all_estimated_cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    
    /*Not visualize an empty pointcloud*/
    pcl::PointXYZRGB noEmptyCloudVisualization;
    noEmptyCloudVisualization.x = 0;
    noEmptyCloudVisualization.y = 0;
    noEmptyCloudVisualization.z = 0;
    noEmptyCloudVisualization.r = 0;
    noEmptyCloudVisualization.g = 0;
    noEmptyCloudVisualization.b = 0;
    cloud_->points.push_back(noEmptyCloudVisualization);
    all_estimated_cloud_->points.push_back(noEmptyCloudVisualization);
    /*Not visualize an empty pointcloud*/
    
    normals_ = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    all_estimated_normals_ = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    
    /*Not visualize an empty pointcloud*/
    pcl::Normal noEmptyNormalVisualization;
    noEmptyNormalVisualization.normal_x = 1;
    noEmptyNormalVisualization.normal_y = 0;
    noEmptyNormalVisualization.normal_z = 0;
    normals_->points.push_back(noEmptyNormalVisualization);
    all_estimated_normals_->points.push_back(noEmptyNormalVisualization);
    /*Not visualize an empty pointcloud*/
    
    rgb_ = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>::Ptr(
        new pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud_));
    all_estimated_rgb_ = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>::Ptr(
        new pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(all_estimated_cloud_));
    
    viewer_->addPointCloud<pcl::PointXYZRGB>(cloud_, *rgb_, "Optimization points");
    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Optimization points");
    viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud_, normals_, 150, 0.35, "normals");
    
    viewer_->addPointCloud<pcl::PointXYZRGB>(all_estimated_cloud_, *all_estimated_rgb_, "Estimated points");
    viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Estimated points");
    viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (all_estimated_cloud_, all_estimated_normals_, 150, 0.35, "estimated normals");
    
    viewer_->setBackgroundColor(0,0,0);
    viewer_->addCoordinateSystem(1.0);
    viewer_->initCameraParameters();
    viewer_->setCameraPose(0,-10,-1,0,0,0,0,-1,-2);
}

void pclVisualizerThread::updateClouds(const std::vector< cv::Vec3d >& pointsVector, const cv::Vec3d &normal, const cv::Scalar &color)
{
    boost::mutex::scoped_lock updateLock(*(updateModelMutex_.get()));
    (*update_) = true;
    
    //     cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_->clear();
    
    /*Not visualize an empty pointcloud*/
    pcl::PointXYZRGB noEmptyCloudVisualization;
    noEmptyCloudVisualization.x = 0;
    noEmptyCloudVisualization.y = 0;
    noEmptyCloudVisualization.z = 0;
    noEmptyCloudVisualization.r = 0;
    noEmptyCloudVisualization.g = 0;
    noEmptyCloudVisualization.b = 0;
    cloud_->points.push_back(noEmptyCloudVisualization);
    /*Not visualize an empty pointcloud*/
    
    //     normals_ = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
    normals_->clear();
    
    /*Not visualize an empty pointcloud*/
    pcl::Normal noEmptyNormalVisualization;
    noEmptyNormalVisualization.normal_x = 1;
    noEmptyNormalVisualization.normal_y = 0;
    noEmptyNormalVisualization.normal_z = 0;
    normals_->points.push_back(noEmptyNormalVisualization);
    /*Not visualize an empty pointcloud*/
    
    for (std::size_t t = 0; t < pointsVector.size(); t++)
    {
        pcl::Normal
        n(normal[0],normal[1],normal[2]);
        normals_->points.push_back(n);
    }
    
    for (int i = 0; i < pointsVector.size(); i++)
    {
        pcl::PointXYZRGB actual;
        actual.x = pointsVector.at(i)[0];
        actual.y = pointsVector.at(i)[1];
        actual.z = pointsVector.at(i)[2];
        actual.r = color[0];
        actual.g = color[1];
        actual.b = color[2];
        
        cloud_->points.push_back(actual);
    }
    cloud_->width = (int) cloud_->points.size ();
    cloud_->height = 1;
    
    updateLock.unlock();
}

void pclVisualizerThread::keepLastCloud()
{
    boost::mutex::scoped_lock updateLock(*(updateModelMutex_.get()));
    (*update_) = true;
    
    for(std::size_t i = 0; i < cloud_->points.size(); i++)
    {
        all_estimated_cloud_->points.push_back(cloud_->points[i]);
        all_estimated_normals_->points.push_back(normals_->points[i]);
    }
    
    updateLock.unlock();
}


void pclVisualizerThread::operator()()
{
    // prepare visualizer named "viewer"
    while (!viewer_->wasStopped ())
    {
        viewer_->spinOnce (100);
        
        // Get lock on the boolean update and check if cloud was updated
        boost::mutex::scoped_lock updateLock(*(updateModelMutex_.get()));
        if((*update_))
        {
            if(!viewer_->updatePointCloud(cloud_, "Optimization points"))
            {
                viewer_->addPointCloud<pcl::PointXYZRGB>(cloud_, *rgb_, "Optimization points");
                viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Optimization points");
            }
            if(!viewer_->updatePointCloud(all_estimated_cloud_, "Estimated points"))
            {
                viewer_->addPointCloud<pcl::PointXYZRGB>(all_estimated_cloud_, *all_estimated_rgb_, "Estimated points");
                viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Estimated points");
            }
            viewer_->removePointCloud("normals", 0);
            viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud_, normals_, 350, 0.35, "normals");
            viewer_->removePointCloud("estimated normals", 0);
            viewer_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (all_estimated_cloud_, all_estimated_normals_, 350, 0.35, "estimated normals");
            (*update_) = false;
        }
        updateLock.unlock();
        
    }   
} 

// void pclVisualizerThread::close()
// {
//     viewer_->close();
// }


} // namespace MOSAIC