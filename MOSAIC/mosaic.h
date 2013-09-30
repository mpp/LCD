/**
 * \file mosaic.h
 * \Author: Michele Marostica
 * \brief: This class work as a descriptor extractor but compute the keypoints
 *         detection using the triangulator objects that I developed in the 
 *         Triangulator folder
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

#ifndef MOSAIC_H
#define MOSAIC_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "DescriptorsMatcher/descriptorsmatcher.h"
#include "Triangulation/singlecameratriangulator.h"
#include "Triangulation/normaloptimizer.h"
#include "Triangulation/neighborhoodsgenerator.h"
#include "pclvisualizerthread.h"
#include "tools.h"



namespace MOSAIC {
    
class MOSAIC : public cv::DescriptorExtractor
{
public:
    
    /** Setup the descriptor extractor
     * @param fs the file with the settings parameters
     * @param imgA the first image for triangulation
     * @param imgB the second image for triangulation
     * @param tA/B the translation of the pose relative to frame A/B
     * @param rA/B the rotation of the pose relative to frame A/B
     */
    MOSAIC(cv::FileStorage& fs, 
           cv::Mat& imgA, cv::Mat& imgB, 
           const cv::Vec3d& tA, const cv::Vec3d& tB, 
           const cv::Vec3d& rA, const cv::Vec3d& rB,
           const std::string& refName );
    
    virtual ~MOSAIC();
    
//     void setFramesAndPoses(const framePosePackage &fpp);
    
//     void computeDescriptors(std::vector<cv::KeyPoint> &kpts, cv::Mat &descriptors);
    void computeDescriptors(std::vector<cv::KeyPoint> &kpts, cv::Mat &descriptors, cv::Mat &descriptors128, std::vector<cv::Vec3d> &triangulated);

    /** Copied from the SURF code, TODO: remove magic numbers
     */
    virtual int descriptorType() const {return CV_32F;}
    virtual int descriptorSize() const {return 128;}
    
protected:
    /*Dirty jobs goes here*/
    virtual void computeImpl(const cv::Mat& image, std::vector< cv::KeyPoint >& keypoints, cv::Mat& descriptors) const;
    
//     void cleanup();
private:    
    
    /*AVOID DEFAULT CONSTRUCTOR = AND ==*/
    MOSAIC();
    MOSAIC& operator=(const MOSAIC& other);
    bool operator==(const MOSAIC& other);
    
    /*Descriptors parameters*/
    cv::Mat 
        imgA_;                              //> the reference image
    cv::Mat 
        imgB_;                              //> the second image used for triangulation
    cv::Mat 
        descriptors_;                       //> the extracted descriptors
    
    DescriptorsMatcher
        *dm_;                               //> The object that provide the matches in frame A and B
    std::vector< cv::KeyPoint >
        kptsA_, kptsB_;                     //> The keypoints found in frame A and B by dm_
    std::vector< cv::DMatch > 
        matches_;                           //> The matches found by dm_
    double
        NNDR_epsilon_;
    
    /*Triangulation parameters*/
    std::vector<cv::Vec3d>
        triangulated_points_;               //> The triangulated points (3D) by sct_
    SingleCameraTriangulator
        *sct_;                              //> The object that triangulate the matches in frame A and B
    cv::Matx44d
        gAB_;                               //> The transformation matrix from the pose A to the pose B
    std::vector<cv::Mat>
        patches_vector_,
        image_points_vector_;
    
    NormalOptimizer
        *no_;                               //> The object that optimize the normals, from the initial guess, 
                                            //> assuming planar features
    std::vector<cv::Vec3d>
        normals_vector_;                    //> The normals found
    std::vector<cv::Matx44d>
        features_frames_;                   //> The coordinate systems of each feature
    NeighborhoodsGenerator
        *ng_;                               //> The object that generate a reference neigborhood (3D)
    std::vector< std::vector<cv::Vec3d> > 
        neighborhoods_vector_;               //> The neighborhoods computed around each feature point (3D)
    std::vector<cv::Vec3d>
        reference_neighborhood_;             //> The reference neighborhood (3D) on the plane X-Y computed by ng_
    
    /*Miscellaneous parameters*/
    std::vector<cv::Scalar>
        colors_;
    std::string
        reference_frame_name_;
    int
        minimum_number_of_matches_;
};

} // namespace MOSAIC

#endif // MOSAIC_H
