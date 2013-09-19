/** 
 * \file neighborhoodsgenerator.h
 * \Author: Michele Marostica
 * \brief: This class has the aim to compute the neighborhoods of a set of 3D points
 *  
 * Copyright (c) 2012, Michele Marostica (michelemaro AT gmail DOT com)
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

#ifndef NEIGHBORHOODSGENERATOR_H
#define NEIGHBORHOODSGENERATOR_H

#include <opencv2/opencv.hpp>

namespace MOSAIC {
    
/** Ray
 */
typedef struct ray_
{
    explicit ray_(double r):r_(r) {};
    double r_;
} ray;

/** Theta
 */
typedef struct theta_
{
    explicit theta_(double t):t_(t) {};
    double t_;
} theta;

/** Sin(theta)
 */
typedef struct sinTheta_
{
    inline explicit sinTheta_(theta t) {st_ = sin(t.t_);};
    double st_;
} sinTheta;

/** 2*Sin^2(theta/2)
 */
typedef struct sinTheta2_
{
    inline explicit sinTheta2_(theta t) { st2_ = 2 * (sin(t.t_/2)) * (sin(t.t_/2)); };
    double st2_;
} sinTheta2;

typedef struct precomputedValue_
{
    explicit precomputedValue_(ray r, theta t, sinTheta st, sinTheta2 st2):
    r_(r), t_(t), st_(st), st2_(st2) {};
    ray r_;
    theta t_;
    sinTheta st_;
    sinTheta2 st2_;
} precomputedValue;

class NeighborhoodsGenerator
{
public:
    NeighborhoodsGenerator(const cv::FileStorage settings);
    
    /** Neighborhoods are sampled by a series of points in concentric circumferences
     *  Pass an empty normal mat for the initial guess
     */
    void computeCircularNeighborhoodsByNormals( const cv::Mat &points, cv::Mat &normals, std::vector<cv::Mat> &neighborhoodsVector );
    
    void computeCircularNeighborhoodByNormal( const cv::Vec3d& point, cv::Vec3d& normal, cv::Mat& neighborhood );
    
    /** Neighborhoods are sampled by a series of points in a square
     *  Pass an empty normal mat for the initial guess
     */
    void computeSquareNeighborhoodsByNormals( const std::vector< cv::Matx44d >& featuresFrames, std::vector< std::vector< cv::Vec3d > >& neighborhoodsVector );
    
    void computeSquareNeighborhoodByNormal( const cv::Matx44d& featureFrame, std::vector< cv::Vec3d >& neighborhood );
    
    void getReferenceSquaredNeighborhood(std::vector<cv::Vec3d> &neighborhood);
    
private:
    NeighborhoodsGenerator();       //> No default constructor
    
private:
    std::vector< precomputedValue >
        lookUpTable_;               //> Precalculated values for the neighborhoods computation
    
    double
        epsilon_,            //> Take a neighborhood of epsilon meters around each point (ray if circular, mid edge if square)
        cm_per_pixel_;                  
    
    int
        number_of_angles_,          //> The number of angles to sample the neighborhood in each circumference
        number_of_rays_;            //> The number of circumferences to sample the neighborhood
};

} // namespace MOSAIC

#endif // NEIGHBORHOODSGENERATOR_H