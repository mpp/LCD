/**
 * \file tools.h
 * \Author: Michele Marostica
 * \brief: Some utilities functions
 *  
 * Copyright (c) 2013, Michele Marostica (michelemaro AT gmail DOT com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * 1 - Redistributions of source code must retain the above copyright notice, 
 *     this list of conditions and the following disclaimer.
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
 */

#ifndef TOOLS_H_
#define TOOLS_H_

#include <math.h>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/thread/thread.hpp>

namespace MOSAIC {
    
template <typename T>
std::string NumberToString(T pNumber);

/** Compute camera matrix and distortion coefficients with the parameters in a file settings
 */
void computeCameraMatixAndDistCoeff(const cv::FileStorage &settings, cv::Matx33d &cameraMatrix, cv::Mat &distCoeff);

/** Compose a 4x4 transform matrix from R and T
 */
void composeTransformation(const cv::Matx33d &R, const cv::Vec3d &T, cv::Matx44d &G);

/** Decompose a 4x4 transform matrix to R and T
 */
void decomposeTransformation(const cv::Matx44d &G, cv::Vec3d &R, cv::Vec3d &T);

/** Generate a random color (RGB)
 */
cv::Scalar random_color(cv::RNG &rng);

/** Generate a skew symmetric matrix from a vector
 */
void getSkewMatrix (const cv::Vec3d &vec, cv::Matx33d &skew);

/** Bilinear interpolation of the intensity of a pixel with non integer coordinates
 */
float getBilinearInterpPix32f ( cv::Mat &cv_img, float x, float y );

/** Draw the matches and return the colors used
 */
void drawMatches(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &window, const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, const std::vector<cv::DMatch> &matches, std::vector<cv::Scalar> &colors, const std::vector<bool> outliersMask);

void drawBackProjectedPoints(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Mat> &points, const std::vector< cv::Scalar > &colors);
void drawBackProjectedPoints(const cv::Mat &input, cv::Mat &output, const cv::Mat &points, const cv::Scalar &colors);

/** Draw a point cloud
 */
void viewPointCloud(const cv::Mat &triagulatedPoints, const std::vector< cv::Scalar > &colors);
void viewPointCloud(const std::vector< cv::Vec3d >& pointsGroup);

void viewPointCloud(const std::vector< cv::Vec3d >& pointsGroup, const cv::Vec3d &normal);

/** Draw a point cloud and given normals
 */
void viewPointCloudAndNormals(const std::vector< cv::Vec3d >& triagulatedPoints, pcl::PointCloud< pcl::Normal >::ConstPtr normals, const std::vector< cv::Scalar >& colors);
void viewPointCloudNormalsAndFrames(const std::vector<cv::Vec3d>& triagulatedPoints, pcl::PointCloud< pcl::Normal >::ConstPtr normals, const std::vector< cv::Scalar >& colors, std::vector<cv::Matx44d> &featuresFrames);
void viewPointCloudNeighborhood(const cv::Mat &triagulatedPoints, std::vector< cv::Mat > &neighborhoodsVector, const std::vector< cv::Scalar > &colors);
void viewPointCloudNormalsFramesAndNeighborhood(const std::vector< std::vector< cv::Vec3d > >& neighborhoodsVector, std::vector< cv::Vec3d >& normals, const std::vector< cv::Scalar >& colors, std::vector< cv::Matx44d >& featuresFrames);

void viewPointCloudNormalsFramesNeighborhoodAndGravity(const std::vector< std::vector< cv::Vec3d > >& neighborhoodsVector, std::vector< cv::Vec3d >& normals, const std::vector< cv::Scalar >& colors, std::vector< cv::Matx44d >& featuresFrames, cv::Vec3d &gravity);

void car2sph(const cv::Vec3d &v, double &phi, double &theta);
void sph2car(const double phi, const double theta, cv::Vec3d &v);

} // namespace MOSAIC

#endif // TOOLS_H_