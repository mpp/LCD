/**
 * \file tools.cpp
 * \Author: Michele Marostica
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

#include <ctime>
#include <Eigen/Geometry>

#include "tools.h"

#include <sstream>

namespace MOSAIC {

template <typename T>
std::string NumberToString(T pNumber)
{
    std::ostringstream oOStrStream;
    oOStrStream << pNumber;
    return oOStrStream.str();
}

void computeCameraMatixAndDistCoeff(const cv::FileStorage &settings, cv::Matx33d &cameraMatrix, cv::Mat &distCoeff)
{
    // 4 Compose the camera matrix
    /*        
     * Fx 0  Cx
     * 0  Fx Cy
     * 0  0  1 
     */
    double
        Fx, Fy, Cx, Cy;
    
    settings["CameraSettings"]["Fx"] >> Fx;
    settings["CameraSettings"]["Fy"] >> Fy;
    settings["CameraSettings"]["Cx"] >> Cx;
    settings["CameraSettings"]["Cy"] >> Cy;
    
    cameraMatrix.zeros();
    
    cameraMatrix(0,0) = Fx;
    cameraMatrix(1,1) = Fy;
    cameraMatrix(0,2) = Cx;
    cameraMatrix(1,2) = Cy;
    cameraMatrix(2,2) = 1;
    
    // 5 Compose the distortion coefficients array
    double
        p1, p2, k0, k1, k2 ;
    
    settings["CameraSettings"]["p1"] >> p1;
    settings["CameraSettings"]["p2"] >> p2;
    settings["CameraSettings"]["k0"] >> k0;
    settings["CameraSettings"]["k1"] >> k1;
    settings["CameraSettings"]["k2"] >> k2;
    
    distCoeff = cv::Mat(cv::Size(5,1), CV_64FC1, cv::Scalar(0));
    
    distCoeff.at<double>(0) = k0;
    distCoeff.at<double>(1) = k1;
    distCoeff.at<double>(2) = p1;
    distCoeff.at<double>(3) = p2;
    distCoeff.at<double>(4) = k2;
}

void composeTransformation(const cv::Matx33d &R, const cv::Vec3d &T, cv::Matx44d &G)
{
    // Put R on top left
    G(0,0) = R(0,0); G(0,1) = R(0,1); G(0,2) = R(0,2);
    G(1,0) = R(1,0); G(1,1) = R(1,1); G(1,2) = R(1,2);
    G(2,0) = R(2,0); G(2,1) = R(2,1); G(2,2) = R(2,2);
    
    // Put T' on top right
    G(0,3) = T(0); G(1,3) = T(1); G(2,3) = T(2);
    
    // Put homogeneous variables
    G(3,0) = 0; G(3,1) = 0; G(3,2) = 0; G(3,3) = 1;
}

void decomposeTransformation(const cv::Matx44d &G, cv::Vec3d &r, cv::Vec3d &t)
{
    cv::Matx33d
        R;
    // Take R from top left
    R(0,0) = G(0,0); R(0,1) = G(0,1); R(0,2) = G(0,2);
    R(1,0) = G(1,0); R(1,1) = G(1,1); R(1,2) = G(1,2);
    R(2,0) = G(2,0); R(2,1) = G(2,1); R(2,2) = G(2,2);
    
    cv::Rodrigues(R, r);
    
    // Take T from top right
    t(0) = G(0,3); t(1) = G(1,3); t(2) = G(2,3);
}

cv::Scalar random_color(cv::RNG &rng)
{
    int color = rng.next();
    return CV_RGB(color&255, (color>>8)&255, (color>>16)&255);
}

void getSkewMatrix (const cv::Vec3d &vec, cv::Matx33d &skew)
{
    skew(0,0) =  0;      skew(0,1) = -vec[2]; skew(0,2) =  vec(1); 
    skew(1,0) =  vec[2]; skew(1,1) =  0;      skew(1,2) = -vec[0]; 
    skew(2,0) = -vec[1]; skew(2,1) =  vec[0]; skew(2,2) =  0; 
}

float getBilinearInterpPix32f ( cv::Mat &cv_img, float x, float y )
{
    int x0 = floor ( double ( x ) ), y0 = floor ( double ( y ) );
    int x1 = x0 + 1, y1 = y0 + 1;
    
    float bilienar_mat[] = { float ( cv_img.at<uchar> ( y0, x0 ) ), float ( cv_img.at<uchar> ( y1, x0 ) ),
        float ( cv_img.at<uchar> ( y0, x1 ) ), float ( cv_img.at<uchar> ( y1, x1 ) )
    };
    float x_mat[] = { 1.0f- ( x- ( float ) x0 ) , ( x- ( float ) x0 ) };
    float y_mat[] = { 1.0f- ( y- ( float ) y0 ) , ( y- ( float ) y0 ) };
    
    return x_mat[0]* ( bilienar_mat[0]*y_mat[0] + bilienar_mat[1]*y_mat[1] ) +
           x_mat[1]* ( bilienar_mat[2]*y_mat[0] + bilienar_mat[3]*y_mat[1] );
}

/** Draw the matches and return the colors used
 */
void drawMatches(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &window, const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, const std::vector<cv::DMatch> &matches, std::vector<cv::Scalar> &colors, const std::vector<bool> outliersMask)
{
    window = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3, cv::Scalar(0));
    
    cv::Mat
        img1BGR, img2BGR;
    
    cv::cvtColor(img1, img1BGR, CV_GRAY2BGR);
    cv::cvtColor(img2, img2BGR, CV_GRAY2BGR);
    
    img1BGR.copyTo(window(cv::Rect(0,0,img1.cols,img1.rows)));
    img2BGR.copyTo(window(cv::Rect(img1.cols,0,img2.cols,img2.rows)));
    
    cv::RNG 
        rng(0xFFF0FF0F);
    
    for (int i = 0; i < matches.size(); i++)
    {
        if (outliersMask[i])
        {
            cv::Scalar 
                color = random_color(rng);
            colors.push_back(color);
            
            int
                idx1 = matches.at(i).queryIdx,
                idx2 = matches.at(i).trainIdx;
                
            cv::Point2f
                pt1 = kpts1.at(idx1).pt,
                pt2 = kpts2.at(idx2).pt;
                
            pt2.x = pt2.x + img1.cols;
            
            cv::circle(window, pt1, 4, color);
            cv::circle(window, pt2, 4, color);
            
            cv::line(window, pt1, pt2, color);
        }
        //         else // All vectors must be N size!!
        //         {
        //             cv::Scalar color(0,0,0);
        //             colors.push_back(color);
        //         }
    }
}

void drawBackProjectedPoints(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Mat> &points, const std::vector< cv::Scalar > &colors)
{
    cv::cvtColor(input, output, CV_GRAY2BGR);
    
    //     output.at<cv::Vec3b>(1,1) = cv::Vec3b(255,0,0);
    
    for (std::size_t i = 0; i < points.size(); i++)
    {
        //         std::cout << points.at(i) << std::endl;
        //         std::cout << points.at(i).rows << " - " << points.at(i).cols << std::endl;
        
        for (std::size_t k = 0; k < points.at(i).rows; k++)
        {
            cv::Point2d
                point(points.at(i).at<cv::Vec2d>(k));
            
            cv::Point2i
                pixel(round(point.x),round(point.y));
                
            // Check for bad points
            if ( !(pixel.x < 0) && !(pixel.x > ( output.cols)) &&
                !(pixel.y < 0) && !(pixel.y > ( output.rows)) )
            {
                //             std::cout << point << " - " << pixel << " - " << colors.at(i)[0] << " " << colors.at(i)[1] << " " << colors.at(i)[2] << std::endl;
                
                output.at<cv::Vec3b>(pixel) = cv::Vec3b(colors.at(i)[0], colors.at(i)[1], colors.at(i)[2]);
            }
        }
    }
}

void drawBackProjectedPoints(const cv::Mat& input, cv::Mat& output, const cv::Mat& points, const cv::Scalar& colors)
{
    output = input;
    
    for (std::size_t k = 0; k < points.rows; k++)
    {
        cv::Point2d
            point = points.at<cv::Vec2d>(k);
        
        cv::Point2i
            pixel(round(point.x),round(point.y));
        
        //      std::cout << point << " - " << pixel << " - " << colors.at(i)[0] << " " << colors.at(i)[1] << " " << colors.at(i)[2] << std::endl;
        
        output.at<cv::Vec3b>(pixel) = cv::Vec3b(colors[0], colors[1], colors[2]);
    }
}


void viewPointCloud(const cv::Mat &triagulatedPoints, const std::vector< cv::Scalar > &colors)
{
    ///////////////////////////// 
    // Converto i punti in point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int i = 0; i < triagulatedPoints.cols; i++)
    {
        // Outliers are already out
        //         if ( outliersMask[i] ) // Do not print outliers
        //         {
        pcl::PointXYZRGB actual;
        actual.x = triagulatedPoints.at<double>(0, i);
        actual.y = triagulatedPoints.at<double>(1, i);
        actual.z = triagulatedPoints.at<double>(2, i);
        actual.r = colors.at(i)[2];
        actual.g = colors.at(i)[1];
        actual.b = colors.at(i)[0];
        
        cloud->points.push_back(actual);
        //         }
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Triangulated points");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
}

void viewPointCloud(const std::vector< cv::Vec3d >& pointsGroup)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int i = 0; i < pointsGroup.size(); i++)
    {
        pcl::PointXYZRGB actual;
        actual.x = pointsGroup.at(i)[0];
        actual.y = pointsGroup.at(i)[1];
        actual.z = pointsGroup.at(i)[2];
        actual.r = 190;
        actual.g = 190;
        actual.b = 255;
        
        cloud->points.push_back(actual);
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Triangulated points");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
}

void viewPointCloud(const std::vector< cv::Vec3d >& pointsGroup, const cv::Vec3d &normal)
{
    pcl::PointCloud<pcl::Normal>::Ptr
        normals(new pcl::PointCloud<pcl::Normal>);
    
    for (std::size_t t = 0; t < pointsGroup.size(); t++)
    {
        pcl::Normal
            n(normal[0],normal[1],normal[2]);
        normals->points.push_back(n);
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int i = 0; i < pointsGroup.size(); i++)
    {
        pcl::PointXYZRGB actual;
        actual.x = pointsGroup.at(i)[0];
        actual.y = pointsGroup.at(i)[1];
        actual.z = pointsGroup.at(i)[2];
        actual.r = 190;
        actual.g = 190;
        actual.b = 255;
        
        cloud->points.push_back(actual);
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Triangulated points");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 150, 0.35, "normals");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPose(0,-8,-1,0,0,0,0,-1,-2);
    
    std::clock_t start;
    double duration;
    
    start = std::clock();
    
    /* Your algorithm here */
    
    
    while (!viewer->wasStopped())
    {
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        viewer->spinOnce();
        if (duration >= 1)
        {
            break;
        }
    }
}

void viewPointCloudAndNormals(const std::vector<cv::Vec3d>& triagulatedPoints, pcl::PointCloud< pcl::Normal >::ConstPtr normals, const std::vector< cv::Scalar >& colors)
{    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    int count = 0;
    for (std::vector<cv::Vec3d>::const_iterator it = triagulatedPoints.begin(); it != triagulatedPoints.end(); it++)
    {
        // Outliers are already out
        //         if ( outliersMask[i] ) // Do not print outliers
        //         {
        pcl::PointXYZRGB actual;
        actual.x = (*it)[0];
        actual.y = (*it)[1];
        actual.z = (*it)[2];
        actual.r = colors.at(count)[2];
        actual.g = colors.at(count)[1];
        actual.b = colors.at(count)[0];
        count++;
        
        cloud->points.push_back(actual);
        //         }
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Triangulated points");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 1, 0.35, "normals");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPose(0,-8,-1,0,0,0,0,-1,-2);
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
    
}

void viewPointCloudNormalsAndFrames(const std::vector<cv::Vec3d>& triagulatedPoints, pcl::PointCloud< pcl::Normal >::ConstPtr normals, const std::vector< cv::Scalar >& colors, std::vector<cv::Matx44d> &featuresFrames)
{    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    int count = 0;
    for (std::vector<cv::Vec3d>::const_iterator it = triagulatedPoints.begin(); it != triagulatedPoints.end(); it++)
    {
        // Outliers are already out
        //         if ( outliersMask[i] ) // Do not print outliers
        //         {
        pcl::PointXYZRGB actual;
        actual.x = (*it)[0];
        actual.y = (*it)[1];
        actual.z = (*it)[2];
        actual.r = colors.at(count)[2];
        actual.g = colors.at(count)[1];
        actual.b = colors.at(count)[0];
        count++;
        
        cloud->points.push_back(actual);
        //         }
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Triangulated points");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 1, 0.35, "normals");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPose(0,-8,-1,0,0,0,0,-1,-2);
    
    // Add frames
    std::string id;
    count = 0;
    std::vector<cv::Matx44d>::iterator f_it = featuresFrames.begin();
    std::vector<cv::Vec3d>::const_iterator p_it = triagulatedPoints.begin();
    while ( f_it != featuresFrames.end() && p_it != triagulatedPoints.end() )
    {
        //         Eigen::Affine3f T;
        //         
        //         T.matrix() << (*f_it)(0,0), (*f_it)(0,1), (*f_it)(0,2), (*f_it)(0,3),
        //                       (*f_it)(1,0), (*f_it)(1,1), (*f_it)(1,2), (*f_it)(1,3),
        //                       (*f_it)(2,0), (*f_it)(2,1), (*f_it)(2,2), (*f_it)(2,3),
        //                       (*f_it)(3,0), (*f_it)(3,1), (*f_it)(3,2), (*f_it)(3,3);
        //         
        //         viewer->addCoordinateSystem(0.3, T);
        // Try use addArrow instead of addFrame 
        pcl::PointXYZ
            base((*p_it)[0],(*p_it)[1],(*p_it)[2]), // The frame start on the feature point
            xP((*f_it)(0,0),(*f_it)(1,0),(*f_it)(2,0)),
            yP((*f_it)(0,1),(*f_it)(1,1),(*f_it)(2,1)),
            zP((*f_it)(0,2),(*f_it)(1,2),(*f_it)(2,2));
        
        // Move arrows to base point
        xP.x = xP.x + base.x; xP.y = xP.y + base.y; xP.z = xP.z + base.z;
        yP.x = yP.x + base.x; yP.y = yP.y + base.y; yP.z = yP.z + base.z;
        zP.x = zP.x + base.x; zP.y = zP.y + base.y; zP.z = zP.z + base.z;
        
        id = "-" + NumberToString<int>(count);
        
        viewer->addArrow(base, xP, 1.0, 0, 0, false, id+"x");
        viewer->addArrow(base, yP, 0, 1.0, 0, false, id+"y");
        viewer->addArrow(base, zP, 0, 0, 1.0, false, id+"z");
        
        f_it++; p_it++; count++;
    }
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
    
}

void viewPointCloudNeighborhood(const cv::Mat &triagulatedPoints, std::vector< cv::Mat > &neighborhoodsVector, const std::vector< cv::Scalar > &colors)
{
    ///////////////////////////// 
    // Converto i punti in point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int i = 0; i < triagulatedPoints.cols; i++)
    {
        pcl::PointXYZRGB actual;
        actual.x = triagulatedPoints.at<double>(0, i);
        actual.y = triagulatedPoints.at<double>(1, i);
        actual.z = triagulatedPoints.at<double>(2, i);
        actual.r = colors.at(i)[2];
        actual.g = colors.at(i)[1];
        actual.b = colors.at(i)[0];
        
        cloud->points.push_back(actual);
        
        for (std::size_t l = 0; l < neighborhoodsVector[i].cols; l++)
        {
            pcl::PointXYZRGB actualNeighbor;
            
            cv::Mat
                neighborhood = neighborhoodsVector[i];
            
            cv::Vec3d
                point = neighborhood.at<cv::Vec3d>(l);
            
            actualNeighbor.x = point[0];
            actualNeighbor.y = point[1];
            actualNeighbor.z = point[2];
            actualNeighbor.r = colors.at(i)[2];
            actualNeighbor.g = colors.at(i)[1];
            actualNeighbor.b = colors.at(i)[0];
            
            cloud->points.push_back(actualNeighbor);
        }
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Feature points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Triangulated points");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPose(0,-8,-1,0,0,0,0,-1,-2);
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
}

void viewPointCloudNormalsFramesAndNeighborhood(const std::vector< std::vector< cv::Vec3d > >& neighborhoodsVector, std::vector<cv::Vec3d> &normals, const std::vector< cv::Scalar >& colors, std::vector< cv::Matx44d >& featuresFrames)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::PointCloud<pcl::Normal>::Ptr
        normalsCloud (new pcl::PointCloud<pcl::Normal>());
    std::vector<cv::Vec3d>::iterator normal_it = normals.begin();
    
    int count = 0;
    for (std::vector< std::vector< cv::Vec3d > >::const_iterator n_it = neighborhoodsVector.begin(); n_it != neighborhoodsVector.end(); n_it++)
    {
        for (std::vector<cv::Vec3d>::const_iterator it = (*n_it).begin(); it != (*n_it).end(); it++)
        {
            pcl::PointXYZRGB actual;
            actual.x = (*it)[0];
            actual.y = (*it)[1];
            actual.z = (*it)[2];
            actual.r = colors.at(count)[2];
            actual.g = colors.at(count)[1];
            actual.b = colors.at(count)[0];
            
            cloud->points.push_back(actual);
            
            
            pcl::Normal
            normal((*normal_it)[0],(*normal_it)[1],(*normal_it)[2]);
            normalsCloud->points.push_back(normal);
        }
        count++;
        normal_it++;
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Triangulated points");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normalsCloud, 1700, 0.15, "normals");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPose(0,-8,-1,0,0,0,0,-1,-2);
    
    // Add frames
    std::string id;
    count = 0;
    std::vector<cv::Matx44d>::iterator f_it = featuresFrames.begin();
    while ( f_it != featuresFrames.end() )
    {
        //         Eigen::Affine3f T;
        //         
        //         T.matrix() << (*f_it)(0,0), (*f_it)(0,1), (*f_it)(0,2), (*f_it)(0,3),
        //         (*f_it)(1,0), (*f_it)(1,1), (*f_it)(1,2), (*f_it)(1,3),
        //         (*f_it)(2,0), (*f_it)(2,1), (*f_it)(2,2), (*f_it)(2,3),
        //         (*f_it)(3,0), (*f_it)(3,1), (*f_it)(3,2), (*f_it)(3,3);
        //         
        //         viewer->addCoordinateSystem(0.3, T);
        
        // Try use addArrow instead of addFrame 
        pcl::PointXYZ
            base((*f_it)(0,3),(*f_it)(1,3),(*f_it)(2,3)), // The frame start on the feature point
            xP((*f_it)(0,0),(*f_it)(1,0),(*f_it)(2,0)),
            yP((*f_it)(0,1),(*f_it)(1,1),(*f_it)(2,1)),
            zP((*f_it)(0,2),(*f_it)(1,2),(*f_it)(2,2));
        
        // Move arrows to base point
        xP.x = xP.x + base.x; xP.y = xP.y + base.y; xP.z = xP.z + base.z;
        yP.x = yP.x + base.x; yP.y = yP.y + base.y; yP.z = yP.z + base.z;
        zP.x = zP.x + base.x; zP.y = zP.y + base.y; zP.z = zP.z + base.z;
        
        id = "-" + NumberToString<int>(count);
        
        viewer->addArrow(base, xP, 1.0, 0, 0, false, id+"x");
        viewer->addArrow(base, yP, 0, 1.0, 0, false, id+"y");
        viewer->addArrow(base, zP, 0, 0, 1.0, false, id+"z");
        
        f_it++; count++;
    }
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
    
}


void viewPointCloudNormalsFramesNeighborhoodAndGravity(const std::vector< std::vector< cv::Vec3d > >& neighborhoodsVector, std::vector< cv::Vec3d >& normals, const std::vector< cv::Scalar >& colors, std::vector< cv::Matx44d >& featuresFrames, cv::Vec3d& gravity)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::PointCloud<pcl::Normal>::Ptr
        normalsCloud (new pcl::PointCloud<pcl::Normal>());
    std::vector<cv::Vec3d>::iterator normal_it = normals.begin();
    
    int count = 0;
    for (std::vector< std::vector< cv::Vec3d > >::const_iterator n_it = neighborhoodsVector.begin(); n_it != neighborhoodsVector.end(); n_it++)
    {
        for (std::vector<cv::Vec3d>::const_iterator it = (*n_it).begin(); it != (*n_it).end(); it++)
        {
            pcl::PointXYZRGB actual;
            actual.x = (*it)[0];
            actual.y = (*it)[1];
            actual.z = (*it)[2];
            actual.r = colors.at(count)[2];
            actual.g = colors.at(count)[1];
            actual.b = colors.at(count)[0];
            
            cloud->points.push_back(actual);
            
            
            pcl::Normal
                normal((*normal_it)[0],(*normal_it)[1],(*normal_it)[2]);
            normalsCloud->points.push_back(normal);
        }
        count++;
        normal_it++;
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
    viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Triangulated points");
    viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normalsCloud, 1700, 0.15, "normals");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPose(0,-8,-1,0,0,0,0,-1,-2);
    
    // Add frames
    std::string id;
    count = 0;
    std::vector<cv::Matx44d>::iterator f_it = featuresFrames.begin();
    while ( f_it != featuresFrames.end() )
    {
        //         Eigen::Affine3f T;
        //         
        //         T.matrix() << (*f_it)(0,0), (*f_it)(0,1), (*f_it)(0,2), (*f_it)(0,3),
        //         (*f_it)(1,0), (*f_it)(1,1), (*f_it)(1,2), (*f_it)(1,3),
        //         (*f_it)(2,0), (*f_it)(2,1), (*f_it)(2,2), (*f_it)(2,3),
        //         (*f_it)(3,0), (*f_it)(3,1), (*f_it)(3,2), (*f_it)(3,3);
        //         
        //         viewer->addCoordinateSystem(0.3, T);
        
        // Try use addArrow instead of addFrame 
        pcl::PointXYZ
            base((*f_it)(0,3),(*f_it)(1,3),(*f_it)(2,3)), // The frame start on the feature point
            xP((*f_it)(0,0),(*f_it)(1,0),(*f_it)(2,0)),
            yP((*f_it)(0,1),(*f_it)(1,1),(*f_it)(2,1)),
            zP((*f_it)(0,2),(*f_it)(1,2),(*f_it)(2,2)),
            g(gravity[0], gravity[1], gravity[2]);
        
        // Move arrows to base point
        xP.x = xP.x + base.x; xP.y = xP.y + base.y; xP.z = xP.z + base.z;
        yP.x = yP.x + base.x; yP.y = yP.y + base.y; yP.z = yP.z + base.z;
        zP.x = zP.x + base.x; zP.y = zP.y + base.y; zP.z = zP.z + base.z;
        g.x = g.x + base.x; g.y = g.y + base.y; g.z = g.z + base.z;
        
        id = "-" + NumberToString<int>(count);
        
        viewer->addArrow(base, xP, 1.0, 0, 0, false, id+"x");
        viewer->addArrow(base, yP, 0, 1.0, 0, false, id+"y");
        viewer->addArrow(base, zP, 0, 0, 1.0, false, id+"z");
        viewer->addArrow(base, g, 1.0, 1.0, 1.0, false, id+"g");
        
        f_it++; count++;
    }
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
    
}

void car2sph(const cv::Vec3d &v, double &phi, double &theta)
{
    theta = atan2( v[2], sqrt(v[0]*v[0] + v[1]*v[1]));
    phi   = atan2( v[1], v[0] );
}
void sph2car(const double phi, const double theta, cv::Vec3d &v)
{
    v[0] = cos( theta ) * cos( phi );
    v[1] = cos( theta ) * sin( phi );
    v[2] = sin( theta );
}

} // namespace MOSAIC