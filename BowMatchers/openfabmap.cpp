/**
 * \file openfabmap.cpp
 * \Author: Michele Marostica
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
#ifndef OPENCV2P4
#define OPENCV2P4
#endif

#include "openfabmap.h"

namespace LCD {
    
    OpenFABMap::OpenFABMap(cv::FileStorage &settings) :
        BoWMatcher< cv::of2::IMatch >()
    {
        // Create detector and extractor
        generateDetector(settings);
        
        if(!feature_detector_) 
        {
            std::cerr << "Feature Detector error" << std::endl;
            exit(-1002);
        }
        
        generateExtractor(settings);
        
        if(!descriptor_extractor_) 
        {
            std::cerr << "Feature Extractor error" << std::endl;
            exit(-1003);
        }
        
        cv::FileStorage 
            fs;
        
        //load FabMap training data
        std::string 
            fabmapTrainDataPath = settings["FilePaths"]["TrainImagDesc"],
            chowliutreePath = settings["FilePaths"]["ChowLiuTree"],
            vocabPath = settings["FilePaths"]["Vocabulary"];
        
        std::cout << "Loading FabMap Training Data" << std::endl;
        
        fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
        
        cv::Mat 
            fabmapTrainData;
        
        fs["BOWImageDescs"] >> fabmapTrainData;
        
        if (fabmapTrainData.empty()) 
        {
            std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found" << std::endl;
            exit(-1004);
        }
        fs.release();
        
        //load a chow-liu tree
        std::cout << "Loading Chow-Liu Tree" << std::endl;
        
        fs.open(chowliutreePath, cv::FileStorage::READ);
        
        fs["ChowLiuTree"] >> cl_tree_;
        
        if (cl_tree_.empty())
        {
            std::cerr << chowliutreePath << ": Chow-Liu tree not found" << std::endl;
            exit(-1005);
        }
        fs.release();
        
        //load the vocabulary
        std::cout << "Loading Vocabulary" << std::endl;
        
        fs.open(vocabPath, cv::FileStorage::READ);
            
        fs["Vocabulary"] >> vocabulary_;
        
        if (vocabulary_.empty())
        {
            std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
            exit(-1006);
        }
        fs.release();
        
        //create options flags
        std::string 
            newPlaceMethod = settings["openFabMapOptions"]["NewPlaceMethod"],
            bayesMethod = settings["openFabMapOptions"]["BayesMethod"];
            
        int 
            simpleMotionModel = settings["openFabMapOptions"]["SimpleMotion"],
            options = 0;
            
        if(newPlaceMethod == "Sampled") 
        {
            options |= cv::of2::FabMap::SAMPLED;
        }
        else
        {
            options |= cv::of2::FabMap::MEAN_FIELD;
        }
        if(bayesMethod == "ChowLiu") 
        {
            options |= cv::of2::FabMap::CHOW_LIU;
        } 
        else 
        {
            options |= cv::of2::FabMap::NAIVE_BAYES;
        }
        if(simpleMotionModel) 
        {
            options |= cv::of2::FabMap::MOTION_MODEL;
        }
        
        //create an instance of the desired type of FabMap
        std::string 
            fabMapVersion = settings["openFabMapOptions"]["FabMapVersion"];
            
        if(fabMapVersion == "FABMAP1") 
        {
            open_fab_map_ = new cv::of2::FabMap1(cl_tree_, 
                                          settings["openFabMapOptions"]["PzGe"],
                                          settings["openFabMapOptions"]["PzGne"],
                                          options,
                                          settings["openFabMapOptions"]["NumSamples"]);
        }
        else if(fabMapVersion == "FABMAPLUT") 
        {
            open_fab_map_ = new cv::of2::FabMapLUT(cl_tree_,
                                            settings["openFabMapOptions"]["PzGe"],
                                            settings["openFabMapOptions"]["PzGne"],
                                            options,
                                            settings["openFabMapOptions"]["NumSamples"],
                                            settings["openFabMapOptions"]["FabMapLUT"]["Precision"]);
        }
        else if(fabMapVersion == "FABMAPFBO") 
        {
            open_fab_map_ = new cv::of2::FabMapFBO(cl_tree_, 
                                            settings["openFabMapOptions"]["PzGe"],
                                            settings["openFabMapOptions"]["PzGne"],
                                            options,
                                            settings["openFabMapOptions"]["NumSamples"],
                                            settings["openFabMapOptions"]["FabMapFBO"]["RejectionThreshold"],
                                            settings["openFabMapOptions"]["FabMapFBO"]["PsGd"],
                                            settings["openFabMapOptions"]["FabMapFBO"]["BisectionStart"],
                                            settings["openFabMapOptions"]["FabMapFBO"]["BisectionIts"]);
        } 
        else if(fabMapVersion == "FABMAP2") 
        {
            open_fab_map_ = new cv::of2::FabMap2(cl_tree_, 
                                          settings["openFabMapOptions"]["PzGe"],
                                          settings["openFabMapOptions"]["PzGne"],
                                          options);
        }
        else 
        {
            std::cerr << "Could not identify openFABMAPVersion from settings file" << std::endl;
            exit(-1001);
        }
        
        //add the training data for use with the sampling method
        open_fab_map_->addTraining(fabmapTrainData);
    }
    
    
//     void OpenFABMap::generateDetector(cv::FileStorage &fs) {
//         
//         //create common feature detector and descriptor extractor
//         std::string 
//             detectorMode = fs["FeatureOptions"]["DetectorMode"],
//             detectorType = fs["FeatureOptions"]["DetectorType"];
//             
//         feature_detector_ = NULL;
//         
//         if(detectorMode == "ADAPTIVE") 
//         {
//             
//             if(detectorType != "STAR" && detectorType != "SURF" && detectorType != "FAST") 
//             {
//                 std::cerr << "Adaptive Detectors only work with STAR, SURF "
//                 "and FAST" << std::endl;
//             }
//             else 
//             {
//                 feature_detector_ = new cv::DynamicAdaptedFeatureDetector(cv::AdjusterAdapter::create(detectorType),
//                                                                  fs["FeatureOptions"]["Adaptive"]["MinFeatures"], 
//                                                                  fs["FeatureOptions"]["Adaptive"]["MaxFeatures"], 
//                                                                  fs["FeatureOptions"]["Adaptive"]["MaxIters"]
//                                                                 );
//             }
//         } 
//         else if(detectorMode == "STATIC") 
//         {
//             if(detectorType == "STAR") 
//             {
//                 
//                 feature_detector_ = new cv::StarFeatureDetector(
//                     fs["FeatureOptions"]["StarDetector"]["MaxSize"],
//                     fs["FeatureOptions"]["StarDetector"]["Response"],
//                     fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
//                     fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
//                     fs["FeatureOptions"]["StarDetector"]["Suppression"]);
//                 
//             }
//             else if(detectorType == "FAST")
//             {
//                 
//                 feature_detector_ = new cv::FastFeatureDetector(
//                     fs["FeatureOptions"]["FastDetector"]["Threshold"],
//                     (int)fs["FeatureOptions"]["FastDetector"]
//                     ["NonMaxSuppression"] > 0);     
//                 
//             } 
//             else if(detectorType == "SURF") 
//             {
//             
//                 #ifdef OPENCV2P4
//                 feature_detector_ = new cv::SURF(
//                     fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
//                     fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
//                     fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
//                     (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
//                                         (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
//                 
//                 #else
//                 feature_detector_ = new cv::SurfFeatureDetector(
//                     fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
//                     fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
//                     fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
//                     (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
//                 #endif
//             } 
//             else if(detectorType == "SIFT") 
//             {
//                 #ifdef OPENCV2P4
//                 feature_detector_ = new cv::SIFT(
//                     fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
//                     fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
//                     fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
//                     fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
//                     fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
//                 #else
//                 feature_detector_ = new cv::SiftFeatureDetector(
//                     fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
//                     fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"]);
//                 #endif
//             }
//             else if(detectorType == "MSER") 
//             {
//                 
//                 feature_detector_ = new cv::MserFeatureDetector(
//                     fs["FeatureOptions"]["MSERDetector"]["Delta"],
//                     fs["FeatureOptions"]["MSERDetector"]["MinArea"],
//                     fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
//                     fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
//                     fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
//                     fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
//                     fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
//                     fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
//                     fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
//                 
//             }
//             else 
//             {
//                 std::cerr << "Could not create detector class. Specify detector "
//                 "options in the settings file" << std::endl;
//             }
//         } 
//         else 
//         {
//             std::cerr << "Could not create detector class. Specify detector "
//             "mode (static/adaptive) in the settings file" << std::endl;
//         }
//     }
//     
//     void OpenFABMap::generateExtractor(cv::FileStorage &fs)
//     {
//         std::string 
//             extractorType = fs["FeatureOptions"]["ExtractorType"];
//         
//         descriptor_extractor_ = NULL;
//         
//         if(extractorType == "SIFT") 
//         {
//             #ifdef OPENCV2P4
//                 descriptor_extractor_ = new cv::SIFT(
//                     fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
//                     fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
//                     fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
//                     fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
//                     fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
//             #else
//                 descriptor_extractor_ = new cv::SiftDescriptorExtractor();
//             #endif
//             
//         } 
//         else if(extractorType == "SURF") 
//         {
//             
//             #ifdef OPENCV2P4
//             descriptor_extractor_ = new cv::SURF(
//                 fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
//                 fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
//                 fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
//                 (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
//                                     (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
//                                     
//             #else
//             descriptor_extractor_ = new cv::SurfDescriptorExtractor(
//                                         fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
//                                         fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
//                                         (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
//                                         (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
//             #endif
//         
//         } 
//         else 
//         {
//             std::cerr << "Could not create Descriptor Extractor. Please specify extractor type in settings file" << std::endl;
//         }
//     
//     }

    void OpenFABMap::compare(cv::Mat &frame, std::vector< cv::of2::IMatch > &matches, cv::Mat &bow, std::vector<cv::KeyPoint> &kpts, std::vector< std::vector < int > > &pointIDXOfCLusters, cv::Mat *completeDescriptors)
    {
        /// Get the BoW of the frame
        
        //use a FLANN matcher to generate bag-of-words representations
        cv::Ptr<cv::DescriptorMatcher>
            matcher = cv::DescriptorMatcher::create("FlannBased");
            
        cv::BOWImgDescriptorExtractor 
            bide(descriptor_extractor_, matcher);
        
        bide.setVocabulary(vocabulary_);
        
        feature_detector_->detect(frame, kpts);
        
        bide.compute(frame, kpts, bow, &pointIDXOfCLusters, completeDescriptors);
        
        
        /// TODO: find a way to use the returned descriptors from the bide extractor
        ///       why? because for now it returns a 0x0 matrix, so find out why :(
        cv::SurfDescriptorExtractor 
            extractor;
        
        extractor.compute(frame, kpts, *completeDescriptors);
        
//         std::cout << "#of keypoints: " << kpts.size() << " - Descriptor size: " << completeDescriptors->cols << "x" << completeDescriptors->rows << std::endl;
        
        /// Compare with the map
        open_fab_map_->compare(bow, matches);
    }

    void OpenFABMap::add(cv::Mat &bow)
    {
        open_fab_map_->add(bow);
    }

    
} // namespace LCD