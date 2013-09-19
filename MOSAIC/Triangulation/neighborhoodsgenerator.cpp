/**
 * \file neighborhoodsgenerator.cpp
 * \Author: Michele Marostica
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

#include <math.h>
#define _USE_MATH_DEFINES

#include "neighborhoodsgenerator.h"
#include "../tools.h"

namespace MOSAIC {

NeighborhoodsGenerator::NeighborhoodsGenerator(const cv::FileStorage settings)
{
    std::string method;
    settings["Neighborhoods"]["method"] >> method;
    
    if (method.compare("square") == 0)
    {
        settings["Neighborhoods"]["epsilon"] >> epsilon_;
        settings["Neighborhoods"]["cmPerPixel"] >> cm_per_pixel_; 
    }
    else if (method.compare("circular") == 0)
    {
        settings["Neighborhoods"]["epsilon"] >> epsilon_;
        settings["Neighborhoods"]["thetas"] >> number_of_angles_;
        settings["Neighborhoods"]["rays"] >> number_of_rays_;
        
        double
            rayIncrement = epsilon_ / number_of_rays_,
            thetaIncrement = 2 * M_PI / number_of_angles_;
        
        for (std::size_t i = 1; i <= number_of_rays_; i++)
        {
            for (std::size_t j = 0; j < number_of_angles_; j++)
            {
                double t = j * thetaIncrement;
                
                lookUpTable_.push_back( precomputedValue( ray(i * rayIncrement),
                                                          theta(t),
                                                          sinTheta(theta(t)),
                                                          sinTheta2(theta(t)) ));
            }
        }
    }
    else
    {
        std::cout << "Unsupported method for plane neighborhood extraction" << std::endl;
        exit(-10);
    }
}

void NeighborhoodsGenerator::computeSquareNeighborhoodsByNormals(const std::vector<cv::Matx44d>& featuresFrames,
                                                                 std::vector< std::vector<cv::Vec3d> >& neighborhoodsVector)
{
    std::vector<cv::Vec3d>
        neighborhood;
    
    neighborhoodsVector.clear();
    
    for(std::vector<cv::Matx44d>::const_iterator it = featuresFrames.begin(); it != featuresFrames.end(); it++)
    {
        computeSquareNeighborhoodByNormal((*it), neighborhood);
        
        neighborhoodsVector.push_back(neighborhood);
    }
}

void NeighborhoodsGenerator::computeSquareNeighborhoodByNormal(const cv::Matx44d& featureFrame,
                                                               std::vector<cv::Vec3d> &neighborhood)
{
    int
        numberOfPointsPerEdge = 2 * ((int) floor(epsilon_ / (0.01 * cm_per_pixel_)));
    
    double
        increment = cm_per_pixel_ * 0.01;
        
    neighborhood.clear();
    
    cv::Vec4d
        pointH(-epsilon_, -epsilon_, 0, 1);
    
    cv::Vec3d
        point;
    
    for (int i = 0; i < numberOfPointsPerEdge; i++)
    {
        for (int j = 0; j < numberOfPointsPerEdge; j++)
        {
            pointH[0] = -epsilon_ + increment * i;
            pointH[1] = -epsilon_ + increment * j;
            pointH[2] = 0;
            pointH[3] = 1;
            
            pointH = featureFrame * pointH;
            
            if (pointH[3] != 1)
            {
                pointH = pointH / pointH[3];
            }
            
            point[0] = pointH[0];
            point[1] = pointH[1];
            point[2] = pointH[2];
            
            neighborhood.push_back(point);
        }
    }
}

void NeighborhoodsGenerator::getReferenceSquaredNeighborhood(std::vector< cv::Vec3d >& neighborhood)
{
    int
        numberOfPointsPerEdge = 2 * ((int) floor(epsilon_ / (0.01 * cm_per_pixel_)));
    
    double
        increment = cm_per_pixel_ * 0.01;
    
    neighborhood.clear();
    
    cv::Vec3d
        point;
    
    for (int i = 0; i < numberOfPointsPerEdge; i++)
    {
        for (int j = 0; j < numberOfPointsPerEdge; j++)
        {
            point[0] = -epsilon_ + increment * i;
            point[1] = -epsilon_ + increment * j;
            point[2] = 0;
            
            neighborhood.push_back(point);
        }
    }
}

void NeighborhoodsGenerator::computeCircularNeighborhoodsByNormals(const cv::Mat& points, cv::Mat& normals, 
                                                                   std::vector< cv::Mat >& neighborhoodsVector)
{
    unsigned int
        numberOfPoints = points.cols,
        numberOfSamples = number_of_angles_ * number_of_rays_;
    
    
    // If no normals, use the initial guess to compute them
    if (normals.empty())
    {
        normals = cv::Mat::zeros(cv::Size(numberOfPoints, 3), CV_64FC1);
        
        for (std::size_t actualPoint = 0; actualPoint < numberOfPoints; actualPoint++)
        {
            cv::Vec3d
                point = points.col(actualPoint),
                normal;
            
            normal = point / cv::norm(point);        
            normals.at<double>(0, actualPoint) = normal[0];
            normals.at<double>(1, actualPoint) = normal[1];
            normals.at<double>(2, actualPoint) = normal[2];
        }
    }
    
    // Compute the neighborhoods
    for (std::size_t actualPoint = 0; actualPoint < numberOfPoints; actualPoint++)
    {
        cv::Vec3d
            point = points.col(actualPoint),
            normal = normals.col(actualPoint);
        
        // Compute a perpendicular vector
        cv::Vec3d
            spanner(0,1,-normal[1]/normal[2]); // in this way <spanner, normal> = 0
        
        //         std::cout << "<" << normal << ", " << spanner << "> = " << normal.dot(spanner) << std::endl;
        
        // Rescale to epsilon_ size 
        spanner = spanner / cv::norm(spanner) * epsilon_;
        
        cv::Mat 
            neighborhood = cv::Mat::zeros(cv::Size(numberOfSamples, 1), CV_64FC3);
        
        for (std::size_t actualSample = 0; actualSample < numberOfSamples; actualSample++)
        {
            cv::Matx33d
                W;
            
            getSkewMatrix(normal, W);
            
            cv::Vec3d
                spannedPoint;
            
            // Compute this formula with precomputed values
            //spannedPoint = point + r * epsilon *(spanner + W*spanner*sin(theta) + (2 * (sin(theta/2)) * (sin(theta/2))) * W * W * spanner);
            
            precomputedValue
                pv = lookUpTable_.at(actualSample);
            
            spannedPoint = point + pv.r_.r_ *(spanner + W * spanner * pv.st_.st_ + pv.st2_.st2_ * W * W * spanner);
            
            //             neighborhood.at<double>(0, actualSample) = spannedPoint[0];
            //             neighborhood.at<double>(1, actualSample) = spannedPoint[1];
            //             neighborhood.at<double>(2, actualSample) = spannedPoint[2];
            neighborhood.at<cv::Vec3d>(actualSample) = spannedPoint;
            
            //             std::cout << point << " - " << spannedPoint << std::endl;
        }
        
        neighborhoodsVector.push_back(neighborhood);
    }
    
}

void NeighborhoodsGenerator::computeCircularNeighborhoodByNormal(const cv::Vec3d& point, cv::Vec3d& normal, cv::Mat& neighborhood)
{
    unsigned int
        numberOfSamples = number_of_angles_ * number_of_rays_;
    
    // If no normal, use the initial guess to compute it
    if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
    {
        normal = point / cv::norm(point);        
    }
    
    // Compute a perpendicular vector
    cv::Vec3d
        spanner(0,1,-normal[1]/normal[2]); // in this way <spanner, normal> = 0
    
    //         std::cout << "<" << normal << ", " << spanner << "> = " << normal.dot(spanner) << std::endl;
    
    // Rescale to epsilon_ size 
    spanner = spanner / cv::norm(spanner) * epsilon_;
    
    neighborhood = cv::Mat::zeros(cv::Size(numberOfSamples, 1), CV_64FC3);
    
    for (std::size_t actualSample = 0; actualSample < numberOfSamples; actualSample++)
    {
        cv::Matx33d
            W;
        
        getSkewMatrix(normal, W);
        
        cv::Vec3d
            spannedPoint;
        
        // Compute this formula with precomputed values
        //spannedPoint = point + r * epsilon *(spanner + W*spanner*sin(theta) + (2 * (sin(theta/2)) * (sin(theta/2))) * W * W * spanner);
        precomputedValue
            pv = lookUpTable_.at(actualSample);
        
        spannedPoint = point + pv.r_.r_ *(spanner + W * spanner * pv.st_.st_ + pv.st2_.st2_ * W * W * spanner);
        
        neighborhood.at<cv::Vec3d>(actualSample) = spannedPoint;
    }
}

} // namespace MOSAIC