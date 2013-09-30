/**
 * \file mosaic.cpp
 * \Author: Michele Marostica
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

#include "mosaic.h"

#include <ctime>

namespace MOSAIC {

MOSAIC::MOSAIC::MOSAIC(cv::FileStorage& fs, 
                cv::Mat& imgA, cv::Mat& imgB, 
                const cv::Vec3d& tA, const cv::Vec3d& tB, 
                const cv::Vec3d& rA, const cv::Vec3d& rB,
                const std::string& refName )
{
    // 1 - Set the images
    imgA_ = imgA.clone();
    imgB_ = imgB.clone();
    
    // 2 - Setup the matcher
    dm_ = new DescriptorsMatcher(fs, imgA_, imgB_);
    fs["NNDR"]["epsilon"] >> NNDR_epsilon_;
    
    fs["loopDetectorOptions"]["minNumOfMatch"] >> minimum_number_of_matches_;
    
    // 3 - Set the transformation matrix between poses A and B
    sct_ = new SingleCameraTriangulator(fs);
    sct_->setg12(tA, tB, rA, rB, gAB_);
    
    // 5 - Setup the normal optimzer
    no_ = new NormalOptimizer(fs, sct_);
    
    // 6 - Setup the neighborhood generator
    ng_ = new NeighborhoodsGenerator(fs);
    ng_->getReferenceSquaredNeighborhood(reference_neighborhood_);
    
    /// computation
    // 2 - Compute the matches
    cv::Mat
        desc1, desc2;
    dm_->compareWithNNDR(NNDR_epsilon_, matches_, kptsA_, kptsB_, desc1, desc2);

    // 3 - Compute the triangulation
    std::vector<bool>
        outliersMask;   // Mask to distinguish between inliers (1) and outliers (0) match points. 
    // As example those with negative z are outliers.
    sct_->setKeypoints(kptsA_, kptsB_, matches_);
    sct_->triangulate(triangulated_points_, outliersMask); // 3D points are all inliers! The mask is for the matches
    
    if (triangulated_points_.size() < minimum_number_of_matches_)
    {
        std::cout << "MOSAIC said: bad couple of images" << std::endl;
        return;
    }
    
    // 4 - Visualizzo/salvo i match
    cv::Mat
        window;
    drawMatches(imgA_, imgB_, window, kptsA_, kptsB_, matches_, colors_, outliersMask);
//     cv::imwrite("./matches/" + refName + "_Matches.pgm", window);
    
    // 5 - Compute the normals and the features frames
    no_->setImages(imgA_, imgB_);

#ifdef ENABLE_VISUALIZER_
    std::cout << "resetting and starting the visualizer" << std::endl;
    no_->startVisualizerThread();
#endif
    no_->computeOptimizedNormals(triangulated_points_, normals_vector_, colors_);
#ifdef ENABLE_VISUALIZER_
    no_->stopVisualizerThread();
#endif

    no_->computeFeaturesFrames(triangulated_points_, normals_vector_, features_frames_);
    
    // 6 - Obtain the patches
    sct_->setImages(imgA_, imgB_);
    sct_->projectReferencePointsToImageWithFrames(reference_neighborhood_, features_frames_, patches_vector_, image_points_vector_);
    
    reference_frame_name_ = refName;
}

MOSAIC::~MOSAIC()
{
    std::cout << "Deleting MOSAIC..." << std::endl;
    delete no_;
    delete ng_;
    delete sct_;
    delete dm_;
}

void MOSAIC::computeDescriptors(std::vector< cv::KeyPoint >& kpts, cv::Mat& descriptors, cv::Mat& descriptors128, std::vector<cv::Vec3d> &triangulated)
{
    
    if (triangulated_points_.size() < minimum_number_of_matches_)
    {
        std::cout << "MOSAIC said: bad couple of images" << std::endl;
        return;
    }
    
    triangulated = std::vector<cv::Vec3d>(triangulated_points_.begin(), triangulated_points_.end());
    
    for (std::vector<cv::Mat>::iterator it = image_points_vector_.begin(); it != image_points_vector_.end(); it++)
    {
        cv::KeyPoint kp;
        kp.angle = -1;
        kp.class_id = 0;
        kp.octave = 0;
        kp.response = 1;
        kp.size = patches_vector_[0].rows;
        
        int half = (int)floor(it->rows/2);
        cv::Vec2d coordinates = it->at<cv::Vec2d>(half);
        
        kp.pt.x = coordinates[0];
        kp.pt.y = coordinates[1];
        
        kpts.push_back(kp);
    }
    
    // 7 - Extract the descriptors
    dm_->extractDescriptorsFromPatches(patches_vector_, descriptors);
    dm_->extractDescriptors128FromPatches(patches_vector_, descriptors128);
    
    // 8 - Draw the patches and save the image
    cv::Mat
        imgA_points;
    drawBackProjectedPoints(imgA_, imgA_points, image_points_vector_, colors_);
    
    cv::imwrite("./matches/" + reference_frame_name_ + "_ProjPatches.jpg", imgA_points);
    
//     for (int i = 0; i < patches_vector_.size(); i++)
//     {
//         cv::imwrite("./patches/" + reference_frame_name_ + "_Patch_" + std::to_string(i) + ".pgm", patches_vector_[i]);
//     }
}
    
// MOSAIC::MOSAIC::MOSAIC(const cv::FileStorage &fs)
// {
//     // 2 - Setup the matcher
//     dm_ = new DescriptorsMatcher(fs);
//     fs["NNDR"]["epsilon"] >> NNDR_epsilon_;
//     fs["loopDetectorOptions"]["minNumOfMatch"] >> minimum_number_of_matches_;
//     
//     // 3 - Set the transformation matrix between poses A and B
//     sct_ = new SingleCameraTriangulator(fs);
//     
//     // 5 - Setup the normal optimzer
//     no_ = new NormalOptimizer(fs, sct_);
//     
//     // 6 - Setup the neighborhood generator
//     ng_ = new NeighborhoodsGenerator(fs);
//     ng_->getReferenceSquaredNeighborhood(reference_neighborhood_);
// }

// void MOSAIC::setFramesAndPoses(const framePosePackage& fpp)
// {
//     // 1 - Set the images
//     imgA_ = fpp.imgA.clone();
//     imgB_ = fpp.imgB.clone();
//     
//     dm_->setImages(imgA_, imgB_);
//     
//     sct_->setg12(fpp.tA, fpp.tB, fpp.rA, fpp.rB, gAB_);
//     
// //     std::cout << fpp.tA << " " << fpp.rA << " " << fpp.tB << " " << fpp.rB << std::endl;
//     
//     reference_frame_name_ = fpp.referenceFrameName;
// }

// void MOSAIC::computeDescriptors(std::vector< cv::KeyPoint >& kpts, cv::Mat& descriptors)
// {
//     /// computation
//     // 2 - Compute the matches
//     cv::Mat
//         desc1, desc2;
//     dm_->compareWithNNDR(NNDR_epsilon_, matches_, kptsA_, kptsB_, desc1, desc2);
//     
//     // 3 - Compute the triangulation
//     std::vector<bool>
//         outliersMask;   // Mask to distinguish between inliers (1) and outliers (0) match points. 
//                         // As example those with negative z are outliers.
//     sct_->setKeypoints(kptsA_, kptsB_, matches_);
//     sct_->triangulate(triangulated_points_, outliersMask); // 3D points are all inliers! The mask is for the matches
//     
//     if (triangulated_points_.empty() || triangulated_points_.size() <= minimum_number_of_matches_)
//     {
//         std::cerr << "MOSAIC said: Bad couple of images." << std::endl;
//         cleanup();
//         return;
//     }
//     
//     // 4 - Visualizzo/salvo i match
//     cv::Mat
//         window;
//     drawMatches(imgA_, imgB_, window, kptsA_, kptsB_, matches_, colors_, outliersMask);
//     cv::imwrite("./matches/" + reference_frame_name_ + "_matches.pgm", window);
//     
//     // 5 - Compute the normals and the features frames
//     no_->setImages(imgA_, imgB_);
//     
// //     no_->startVisualizerThread();
//     no_->computeOptimizedNormals(triangulated_points_, normals_vector_, colors_);
// //     no_->stopVisualizerThread();
//     
//     if (triangulated_points_.empty())
//     {
//         std::cerr << "MOSAIC said: Bad couple of images." << std::endl;
//         cleanup();
//         return;
//     }
//     
//     no_->computeFeaturesFrames(triangulated_points_, normals_vector_, features_frames_);
//     
//     // 6 - Obtain the patches
//     sct_->setImages(imgA_, imgB_);
//     sct_->projectReferencePointsToImageWithFrames(reference_neighborhood_, features_frames_, patches_vector_, image_points_vector_);
// 
//     
//     for (std::vector<cv::Mat>::iterator it = image_points_vector_.begin(); it != image_points_vector_.end(); it++)
//     {
//         cv::KeyPoint kp;
//         kp.angle = -1;
//         kp.class_id = 0;
//         kp.octave = 0;
//         kp.response = 1;
//         kp.size = 128;
//         
//         int half = (int)floor(it->rows/2);
//         cv::Vec2d coordinates = it->at<cv::Vec2d>(half);
//         
//         kp.pt.x = coordinates[0];
//         kp.pt.y = coordinates[1];
//         
//         kpts.push_back(kp);
//     }
//     
//     // 7 - Extract the descriptors
//     dm_->extractDescriptorsFromPatches(patches_vector_, descriptors);
//     
//     // 8 - Draw the patches and save the image
//     cv::Mat
//         imgA_points;
//     drawBackProjectedPoints(imgA_, imgA_points, image_points_vector_, colors_);
//     
//     std::string name = reference_frame_name_;
//     if (name.compare("") == true)
//     {
//         name = NumberToString<int>((int)std::time(0));
//     }
//     
//     cv::imwrite("./matches/" + name + "_ProjPatches.pgm", imgA_points);
//     
//     for (int i = 0; i < patches_vector_.size(); i++)
//     {
//         cv::imwrite("./patches/" + name + "_Patch_" + NumberToString<int>(i) + ".pgm", patches_vector_[i]);
//     }
//     
//     // Do the clean-up
//     cleanup();
// }
// 
// void MOSAIC::computeDescriptors(std::vector< cv::KeyPoint >& kpts, cv::Mat& descriptors, cv::Mat& descriptors128)
// {
//     /// computation
//     // 2 - Compute the matches
//     cv::Mat
//         desc1, desc2;
//     dm_->compareWithNNDR(NNDR_epsilon_, matches_, kptsA_, kptsB_, desc1, desc2);
//     
//     // 3 - Compute the triangulation
//     std::vector<bool>
//         outliersMask;   // Mask to distinguish between inliers (1) and outliers (0) match points. 
//     // As example those with negative z are outliers.
//     sct_->setKeypoints(kptsA_, kptsB_, matches_);
//     sct_->triangulate(triangulated_points_, outliersMask); // 3D points are all inliers! The mask is for the matches
//     
//     if (triangulated_points_.empty() || triangulated_points_.size() <= minimum_number_of_matches_)
//     {
//         std::cerr << "MOSAIC said: Bad couple of images." << std::endl;
//         cleanup();
//         return;
//     }
//     
//     // 4 - Visualizzo/salvo i match
//     cv::Mat
//         window;
//     drawMatches(imgA_, imgB_, window, kptsA_, kptsB_, matches_, colors_, outliersMask);
//     cv::imwrite("./matches/" + reference_frame_name_ + "_matches.pgm", window);
//     
//     // 5 - Compute the normals and the features frames
//     no_->setImages(imgA_, imgB_);
//     
// #ifdef ENABLE_VISUALIZER_
//     std::cout << "resetting and starting the visualizer" << std::endl;
//     no_->resetVisualizer();
//     no_->startVisualizerThread();
// #endif
//     no_->computeOptimizedNormals(triangulated_points_, normals_vector_, colors_);
// #ifdef ENABLE_VISUALIZER_
//     no_->stopVisualizerThread();
// #endif
//     
//     no_->computeFeaturesFrames(triangulated_points_, normals_vector_, features_frames_);
//     
//     // 6 - Obtain the patches
//     sct_->setImages(imgA_, imgB_);
//     sct_->projectReferencePointsToImageWithFrames(reference_neighborhood_, features_frames_, patches_vector_, image_points_vector_);
//     
//     
//     for (std::vector<cv::Mat>::iterator it = image_points_vector_.begin(); it != image_points_vector_.end(); it++)
//     {
//         cv::KeyPoint kp;
//         kp.angle = -1;
//         kp.class_id = 0;
//         kp.octave = 0;
//         kp.response = 1;
//         kp.size = 128;
//         
//         int half = (int)floor(it->rows/2);
//         cv::Vec2d coordinates = it->at<cv::Vec2d>(half);
//         
//         kp.pt.x = coordinates[0];
//         kp.pt.y = coordinates[1];
//         
//         kpts.push_back(kp);
//     }
//     
//     // 7 - Extract the descriptors
//     dm_->extractDescriptorsFromPatches(patches_vector_, descriptors);
//     dm_->extractDescriptors128FromPatches(patches_vector_, descriptors128);
//     
//     // 8 - Draw the patches and save the image
//     cv::Mat
//         imgA_points;
//     drawBackProjectedPoints(imgA_, imgA_points, image_points_vector_, colors_);
//     
//     std::string name = reference_frame_name_;
//     if (name.compare("") == true)
//     {
//         name = NumberToString<int>((int)std::time(0));
//     }
//     
//     cv::imwrite("./matches/" + name + "_ProjPatches.pgm", imgA_points);
//     
//     for (int i = 0; i < patches_vector_.size(); i++)
//     {
//         cv::imwrite("./patches/" + name + "_Patch_" + NumberToString<int>(i) + ".pgm", patches_vector_[i]);
//     }
//     
//     // Do the clean-up
//     cleanup();
// }
// 
// void MOSAIC::cleanup()
// {
//     matches_.clear();
//     kptsA_.clear();
//     kptsB_.clear();
//     colors_.clear();
//     triangulated_points_.clear();
//     normals_vector_.clear();
//     features_frames_.clear();
//     patches_vector_.clear();
//     image_points_vector_.clear();
// }


void MOSAIC::computeImpl(const cv::Mat& image, std::vector< cv::KeyPoint >& keypoints, cv::Mat& descriptors) const
{
}
//     /// Usare un metodo setImages per passare frame A e B e ignorare image e keypoints
//     /// verificare cosa passare alla funzione compute affiché non scazzi tutto. DC
//     
//     // 1 - Check the images
//     if (imgA_.empty() || imgB_.empty())
//     {
//         std::cout << "Images not valid" << std::endl;
//         return;
//     }
//     
//     // 7 - Extract the descriptors
//     dm_->extractDescriptorsFromPatches(patches_vector_, descriptors);
//     
//     // 8 - Draw the patches and save the image
//     cv::Mat
//         imgA_points;
//     drawBackProjectedPoints(imgA_, imgA_points, image_points_vector_, colors_);
//     
//     std::string name = reference_frame_name_;
//     if (name.compare("") == true)
//     {
//         name = NumberToString<int>((int)std::time(0));
//     }
//     
//     cv::imwrite(name + "_ProjPatches.pgm", imgA_points);
// }


} // namespace MOSAIC