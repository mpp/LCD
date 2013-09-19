/**
 * \file singlecameratriangulator.h
 * \Author: Michele Marostica
 * \brief: This class take the matched kpts of two images and triangulate them
 *         using the transformation matrix and the camera parameters that are 
 *         passed to the constructor.
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

#ifndef SINGLE_CAMERA_TRIANGULATOR_H_
#define SINGLE_CAMERA_TRIANGULATOR_H_

#include <vector>

#include <opencv2/opencv.hpp>

#include "../tools.h"

namespace MOSAIC {
    
// TODO: maybe it would be easier and more readable if use cv::Point2f and a float
typedef struct
{
    double x_; //> x coordinate of the pixel
    double y_; //> y coordinate of the pixel
    float i_;  //> intensity of the pixel
} Pixel;

enum IMAGE_ID{image1,image2};

class SingleCameraTriangulator
{
public:
    
    SingleCameraTriangulator(const cv::FileStorage &settings );
    
    void setImages(const cv::Mat& img1, const cv::Mat& img2);
    
    void setg12(const cv::Vec3d& T1, const cv::Vec3d& T2, const cv::Vec3d& rodrigues1, const cv::Vec3d& rodrigues2, cv::Matx44d &g12);
    
    void setKeypoints( const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, const std::vector<cv::DMatch> &matches);
    
    void triangulate( std::vector< cv::Vec3d >& triangulatedPoints, std::vector< bool >& outliersMask );
    
    void projectPointsAndComputeResidual( const std::vector< cv::Mat >& pointsGroupVector, std::vector< cv::Mat >& imagePointsVector1, std::vector< cv::Mat >& imagePointsVector2, std::vector< std::vector<double> > &residualsVectors );
    
    void projectPointsAndComputeResidual( const cv::Mat& pointsGroup, cv::Mat& imagePoints1, cv::Mat& imagePoints2, std::vector<double> &residualsVector );
    
    /** Extract a contour of pixels in the image1 and project them in the 3D plane according to the normal
     */
    void extractPixelsContourAndGet3DPoints( const cv::Vec3d &point, const cv::Vec3d &normal, std::vector<Pixel> &pixels, std::vector<cv::Vec3d>& pointsGroup);
    
    void extractPixelsContour( const cv::Vec3d& point, std::vector< Pixel >& pixels );
    
    int get3dPointsFromImage1Pixels(const cv::Vec3d& point, const cv::Vec3d &normal, const cv::Mat &pixelMat, std::vector< cv::Vec3d >& pointsGroup);
    
    inline void projectPointsToImage2( const std::vector<cv::Vec3d> &pointsGroup, std::vector<Pixel> &pixels ) {projectPointsToImage2(pointsGroup, 1.0, pixels);}
    
    int projectPointsToImage2( const std::vector<cv::Vec3d> &pointsGroup, const double scale, std::vector<Pixel> &pixels );
    
    int updateImage1PixelsIntensity(const double scale, std::vector<Pixel> &pixels );
    
    void projectPointsToImage(const IMAGE_ID id, 
                              const std::vector< std::vector< cv::Vec3d > >& pointsGroupVector, 
                              std::vector< cv::Mat >& patchesVector,
                              std::vector< cv::Mat >& imagePointsVector
    );
    
    void projectReferencePointsToImageWithFrames(const std::vector< cv::Vec3d >& referenceNeighborhooh, 
                                                 const std::vector< cv::Matx44d >& featureFrames, 
                                                 std::vector< cv::Mat >& patchesVector, 
                                                 std::vector< cv::Mat >& imagePointsVector );
    //     int getMdat();
    // Private methods
private:
    SingleCameraTriangulator(); // Avoid the default constructor
    
    void extractPixelsContour( const cv::Vec2d &point, std::vector<Pixel> &pixels );
    
    void projectPointsToImages( const std::vector< cv::Mat >& pointsGroupVector, std::vector< cv::Mat >& imagePointsVector1, std::vector< cv::Mat >& imagePointsVector2 );
    
    void projectPointsToImages( const cv::Mat& pointsGroup, cv::Mat& imagePoints1, cv::Mat& imagePoints2 );
    
    void projectPointsToImage(const IMAGE_ID id, 
                              const std::vector< cv::Vec3d >& pointsGroup, 
                              cv::Mat& patch, 
                              cv::Mat& imagePoints );
    
    void projectReferencePointsToImageWithFrame(const std::vector< cv::Vec3d >& referenceNeighborhooh, 
                                                const cv::Matx44d& featureFrame, 
                                                cv::Mat& patch, 
                                                cv::Mat& imagePoints );
    
    void projectPointToPlane( const cv::Vec3d& newPoint, const cv::Vec3d& featurePoint, const cv::Vec3d& normal, cv::Vec3d& pointOnThePlane );
    
    bool isInBoundingBox( const cv::Vec3d &point );
    
    bool isPixelGood( const Pixel& pixel, const double scale );
    // Private data
private:
    
    cv::Ptr<cv::Mat>
        img_1_,
        img_2_;
    
    std::size_t
        N_;                         //> Number of points
    
    cv::Ptr<cv::Matx33d>
        camera_matrix_;             //> Camera Matrix
    
    cv::Ptr<cv::Mat>
        distortion_coefficients_;   //> Camera DistCoeff
    
    cv::Ptr<cv::Mat>
        a_1_,                        //> Keypoints in the first image
        a_2_,                        //> Keypoints in the second image
        u_1_,                        //> Undistorted Keypoints in the first image
        u_2_;                        //> Undistorted Keypoints in the second image
    
//     std::vector< cv::KeyPoint >
//         kpts_1_,                     //> Keypoints in the first image
//         kpts_2_;                     //> Keypoints in the first image

//     std::vector< cv::DMatch > 
//         matches_;                   //> Vector of matches from the first image to the second
    
    cv::Ptr<cv::Vec3d>
        translation_IC_;
    
    cv::Ptr<cv::Matx33d>
        //         rotation_1_,                 //> The R matrix of the first image
        //         rotation_2_,                 //> The R matrix of the second image
        rotation_IC_;                //> The R matrix from the camera frame to the IMU frame
    
    cv::Ptr<cv::Matx44d>
        //         g_1_,                        //> Matrice di trasformazione da CAMERA 1 a E (earth)
        //         g_2_,                        //> Matrice di trasformazione da CAMERA 2 a E (earth)
        g_12_,                       //> Matrice di trasformazione da CAMERA 2 a CAMERA 1
        g_IC_;                       //> gIC = Matrice di trasformazione da CAMERA a IMU
    
    cv::Ptr<cv::Matx34d>
        projection_1_,               //> Camera 1 projection matrix
        projection_2_;               //> Camera 2 projection matrix
    
    std::vector<cv::Vec3d>
    //         triagulated_homogeneous_,   //> Triangulated points in 4x1 homogeneous coordinates
        triagulated_;               //> Triangulated points in 3x1 euclidean coordinates
    
    std::vector<bool>
        outliers_mask_; //> Mask to distinguish between inliers (1) and outliers (0) points. As example those with negative z are outliers.
    
    double
        z_threshold_min_, //> Threshold to define too near outliers 
        z_threshold_max_; //> Threshold to define too far outliers
    
    int
        pixels_ray_;
};

} // namespace MOSAIC

#endif // SINGLE_CAMERA_TRIANGULATOR_H_

