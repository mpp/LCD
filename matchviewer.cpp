/**
 * \file matchviewer.cpp
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

#include <iomanip>
#include "matchviewer.h"

#include <boost/lexical_cast.hpp>
#include <limits>

namespace LCD {
    
    MatchViewer::MatchViewer()
    {
        // Setup matricies
        
        cv::Scalar backgroundColor(200,200,200);
        window_ = cv::Mat(cv::Size(kWindowWidth, kWindowHeight), CV_8UC3, backgroundColor);
        
        actual_frame_ = cv::Mat(cv::Size(kFrameWidth, kFrameHeight), CV_8UC3, cv::Scalar(0));
        
        best_match_frame_ = cv::Mat(cv::Size(kFrameWidth, kFrameHeight), CV_8UC3, cv::Scalar(0));
    
        match_plot_ = cv::Mat(cv::Size(kPlotWidth, kPlotHeight), CV_8UC3, cv::Scalar(0));
        
        path_ = cv::Mat(cv::Size(kPathDimension, kPathDimension), CV_8UC3, cv::Scalar(0));
        
        left_info_display_ = cv::Mat(cv::Size(kDisplayWidth, kDisplayHeight), CV_8UC3, cv::Scalar(0));
        
        right_info_display_ = cv::Mat(cv::Size(kDisplayWidth, kDisplayHeight), CV_8UC3, cv::Scalar(0));
        
        font_alert_ = cv::fontQt("Arial", 12, cv::Scalar(255,0,0));
        
        font_text_ = cv::fontQt("Arial", 12, cv::Scalar(255,255,255));
    }
    
    void MatchViewer::update(cv::Mat& window, int similarFrameMapIndex, bool isMatchAccept,
                             const FramePoseList& list, std::vector< cv::DMatch >& matchArray,
                             std::vector< std::pair<int,int> > &matchesIndices, std::string leftInfoDisplay, std::string rightInfoDisplay)
    {
        FramePose actualFrameElement = *(list.End()-1);
        cv::Mat actualFrame = cv::imread(actualFrameElement.frame_path_, CV_LOAD_IMAGE_COLOR);
        
        cv::Mat similarFrame;
        
        if (similarFrameMapIndex >= 0)
        {
            FramePose similarFrameElement;
            list.getMapFramePoseAt(similarFrameMapIndex, similarFrameElement);
            similarFrame = cv::imread(similarFrameElement.frame_path_, CV_LOAD_IMAGE_COLOR);
            
//             // Draw keypoints
//             std::vector<int>
//             queryIdx, trainIdx;
//             
//             for (std::size_t t = 0; t < matchesIndices.size(); t++)
//             {
//                 queryIdx.push_back(matchesIndices[t].first);
//                 trainIdx.push_back(matchesIndices[t].second);
//             }
//             
//             drawKeyPoints(actualFrame, actualFrameElement.kpts_, actualFrameElement.point_IDX_of_clusters_, queryIdx);
//             drawKeyPoints(similarFrame, similarFrameElement.kpts_, similarFrameElement.point_IDX_of_clusters_, trainIdx);
            
        }
        else
        {
            similarFrame = actualFrame;
        }
        // Resize and add to the window
        cv::resize(actualFrame,actual_frame_,cv::Size(kFrameWidth,kFrameHeight));
        cv::resize(similarFrame,best_match_frame_,cv::Size(kFrameWidth,kFrameHeight));
        
        if (isMatchAccept)
        {
            best_match_frame_ += cv::Scalar(0,20,0);
        }
        else
        {
            best_match_frame_ += cv::Scalar(0,0,20);
        }
        
        infoDisplay(left_info_display_, leftInfoDisplay);
        
        infoDisplay(right_info_display_, rightInfoDisplay);
        
        plotArray(match_plot_, matchArray);
        
        drawPath(path_, list, similarFrameMapIndex, isMatchAccept);
        
        left_info_display_.copyTo(window_(cv::Rect(kLeftDisplayLeft, kDisplayTop, kDisplayWidth, kDisplayHeight)));
        right_info_display_.copyTo(window_(cv::Rect(kRightDisplayLeft, kDisplayTop, kDisplayWidth, kDisplayHeight)));
        
        actual_frame_.copyTo(window_(cv::Rect(kActualFrameLeft, kActualFrameTop, kFrameWidth, kFrameHeight)));
        best_match_frame_.copyTo(window_(cv::Rect(kSimilarFrameLeft, kSimilarFrameTop, kFrameWidth, kFrameHeight)));
        
        match_plot_.copyTo(window_(cv::Rect(kPlotLeft, kPlotTop, kPlotWidth, kPlotHeight)));
        
        path_.copyTo(window_(cv::Rect(kPathLeft, kPathTop, kPathDimension, kPathDimension)));
        
        // the private member is in the right size, the output not.
        window = window_;
    }


    
    void MatchViewer::plotArray(cv::Mat& plot, std::vector< cv::DMatch >& matchArray)
    {
        
        // Normalize the likelihood
        double
            minDistance = std::numeric_limits< double >::max(),
            maxDistance = std::numeric_limits< double >::min();
        
        for (size_t i = 0; i < matchArray.size(); i++)
        {
            if (minDistance > matchArray[i].distance)
            {
                minDistance = matchArray[i].distance;
            }
            if (maxDistance < matchArray[i].distance)
            {
                maxDistance = matchArray[i].distance;
            }
        }
        
        // Reset the plot area
        plot = cv::Scalar(256,256,256);
        
        unsigned int step = std::min<int>(floor((kPlotWidth-10) / (matchArray.size() + 1)), 100); 
        unsigned int line_max_lenght = 200;
        unsigned int line_weight = 1;
        
        // y axis
        cv::line(plot,
                 cv::Point(10,10),
                 cv::Point(10,kPlotHeight - 5),
                 cv::Scalar(0,0,0),
                 1
        );
        cv::line(plot,
                 cv::Point(9,kPlotHeight - 10 - line_max_lenght),
                 cv::Point(11,kPlotHeight - 10 - line_max_lenght),
                 cv::Scalar(0,0,0),
                 1
        );
        std::stringstream ss;
        ss << std::setprecision(2) << maxDistance;
        cv::addText(plot, 
                    ss.str(), 
                    cv::Point(5,kPlotHeight - 10 - line_max_lenght),
                    cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
        );
        cv::addText(plot, 
                    "0", 
                    cv::Point(5,kPlotHeight -3),
                    cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
        );
        
        // x axis
        cv::line(plot,
                 cv::Point(5,kPlotHeight - 10),
                 cv::Point(kPlotWidth - 5,kPlotHeight - 10),
                 cv::Scalar(0,0,0),
                 1
        );
        
        //         std::cout << "Debug: " << matchArray.cols << " " << step << " " << floor(line_max_lenght * matchArray.at<float>(5)) << std::endl;
        
        for (size_t i = 0; i < matchArray.size(); i++)
        {
            // add the line and the number
            cv::line(plot,
                     cv::Point(15 + i*step,246),
                     cv::Point(15 + i*step,248),
                     cv::Scalar(0,0,0),
                     1
            );
            
            double distanceValue = matchArray[i].distance / maxDistance;
            
            cv::line(plot,
                     cv::Point(15 + i*step,246),
                     cv::Point(15 + i*step,246 - floor((line_max_lenght) * distanceValue)),
                     cv::Scalar(150,255,150),
                     2
            );
            
            cv::addText(plot, 
                        boost::lexical_cast<std::string>(matchArray[i].imgIdx), 
                        cv::Point(15 + i*step,253),
                        cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
                        );
            cv::addText(plot, 
                        boost::lexical_cast<std::string>(matchArray[i].distance), 
                        cv::Point(15 + i*step,251 - floor((line_max_lenght) * distanceValue)),
                        cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
                        );
            
        }
        
    }

    
    void MatchViewer::plotArray(cv::Mat& plot, std::vector<cv::of2::IMatch> &matchArray)
    {
        
        // Normalize the likelihood
        double
            minLikelihood = std::numeric_limits< double >::max(),
            maxLikelihood = std::numeric_limits< double >::min(),
            maxLikelihoodValue = 0.0;
        
        for (size_t i = 0; i < matchArray.size(); i++)
        {
            if (minLikelihood > matchArray[i].likelihood)
            {
                minLikelihood = matchArray[i].likelihood;
            }
            if (maxLikelihood < matchArray[i].likelihood)
            {
                maxLikelihood = matchArray[i].likelihood;
            }
        }
            
//         std::cout << "min " << minLikelihood << " - max " << maxLikelihood << std::endl;
        maxLikelihoodValue = std::max<double>(maxLikelihood, -minLikelihood);
        
        // Reset the plot area
        plot = cv::Scalar(256,256,256);
        
        unsigned int step = std::min<int>(floor((kPlotWidth-10) / matchArray.size()), 100); 
        unsigned int line_max_lenght = 200;
        unsigned int line_weight = 1;
        
        // y axis
        cv::line(plot,
                 cv::Point(10,10),
                 cv::Point(10,kPlotHeight - 5),
                 cv::Scalar(0,0,0),
                 1
                );
        cv::line(plot,
                 cv::Point(9,kPlotHeight - 10 - line_max_lenght),
                 cv::Point(11,kPlotHeight - 10 - line_max_lenght),
                 cv::Scalar(0,0,0),
                 1
                );
        cv::addText(plot, 
                    "1", 
                    cv::Point(5,kPlotHeight - 10 - line_max_lenght),
                    cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
                    );
        cv::addText(plot, 
                    "0", 
                    cv::Point(5,kPlotHeight -3),
                    cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
                    );
        
        // x axis
        cv::line(plot,
                 cv::Point(5,kPlotHeight - 10),
                 cv::Point(kPlotWidth - 5,kPlotHeight - 10),
                 cv::Scalar(0,0,0),
                 1
                );
        
        // x axis likelihood
        cv::line(plot,
                 cv::Point(5, kPlotHeight - 10 - floor(line_max_lenght/2) ),
                 cv::Point(kPlotWidth - 5,kPlotHeight - 10 - floor(line_max_lenght/2)),
                 cv::Scalar(0,200,0),
                 1
                );
        
        // y axis likelihood
        cv::line(plot,
                 cv::Point(kPlotWidth - 10, 5),
                 cv::Point(kPlotWidth - 10,kPlotHeight - 5),
                 cv::Scalar(0,200,0),
                 1
                );
        cv::addText(plot, 
                    "1", 
                    cv::Point(kPlotWidth - 8,kPlotHeight - 11 - line_max_lenght),
                    cv::fontQt("Arial", 5, cv::Scalar(0,200,0))
                    );
        cv::addText(plot, 
                    "0", 
                    cv::Point(kPlotWidth - 8,kPlotHeight - 11 - floor(line_max_lenght/2)),
                    cv::fontQt("Arial", 5, cv::Scalar(0,200,0))
                    );
        cv::addText(plot, 
                    "-1", 
                    cv::Point(kPlotWidth - 8,kPlotHeight - 11),
                    cv::fontQt("Arial", 5, cv::Scalar(0,200,0))
                    );
        
        // loop closure accept threshold
        cv::line(plot,
                 cv::Point(5,kPlotHeight - 10 - floor(line_max_lenght * 0.99)),
                 cv::Point(kPlotWidth - 5,kPlotHeight - 10 - floor(line_max_lenght * 0.99)),
                 cv::Scalar(0,0,255),
                 1
                );
        cv::addText(plot, 
                    "Loop closure threshold", 
                    cv::Point(15,kPlotHeight - 11 - floor(line_max_lenght * 0.99)),
                    cv::fontQt("Arial", 5, cv::Scalar(200,0,0))
                    );
        
        
        
//         std::cout << "Debug: " << matchArray.cols << " " << step << " " << floor(line_max_lenght * matchArray.at<float>(5)) << std::endl;
        
        for (size_t i = 0; i < matchArray.size(); i++)
        {
            // add the line and the number
            cv::line(plot,
                     cv::Point(15 + i*step,246),
                     cv::Point(15 + i*step,246 - floor(line_max_lenght * matchArray[i].match)),
                     cv::Scalar(150,150,150),
                     2
                    );
            cv::line(plot,
                     cv::Point(15 + i*step,246),
                     cv::Point(15 + i*step,248),
                     cv::Scalar(0,0,0),
                     1
                    );
            
            double likelihoodValue = matchArray[i].likelihood / maxLikelihoodValue;
            
//             std::cout << i << " - " << likelihoodValue << std::endl;
            
            cv::line(plot,
                     cv::Point(17 + i*step,146),
                     cv::Point(17 + i*step,146 - floor((line_max_lenght / 2) * likelihoodValue)),
                     cv::Scalar(150,255,150),
                     2
                    );
            cv::line(plot,
                     cv::Point(17 + i*step,145),
                     cv::Point(17 + i*step,147),
                     cv::Scalar(200,0,0),
                     1
                    );
            
            if ((i-1)%2 == 0)
            {
                cv::addText(plot, 
                            "#" + boost::lexical_cast<std::string>(i-1), 
                            cv::Point(15 + i*step,253),
                            cv::fontQt("Arial", 5, cv::Scalar(0,0,0))
                );
            }
        }
        
    }

    void MatchViewer::drawPath(cv::Mat& path, const FramePoseList& list, int similarFrameMapIndex, bool isMatchAccept)
    {
        
        path = cv::Scalar(0);
        
        std::vector<FramePose>::const_iterator 
            actualFramePose;
            
        cv::Vec3f
            actualPosisiton,
            oldPosition(0,0,0);
            
        float scale = kPathScale / 100;    
        
        int mapCounter = 0;
        
        for (actualFramePose = list.Begin(); actualFramePose != list.End(); actualFramePose++)
        {
            actualPosisiton = actualFramePose->pose_T_;
            
            cv::line(path, 
                     cv::Point(ceil(scale*oldPosition[0] + kPathTranslationX), ceil(scale*-oldPosition[1] + kPathTranslationY)), 
                     cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                     cv::Scalar(0,0,255),
                     1
            );
            
            if (actualFramePose->is_on_map_)
            {
                cv::Scalar 
                    color;
                    
                if (mapCounter++ == similarFrameMapIndex)
                {
                    if (isMatchAccept == true)
                    {
                        color = cv::Scalar(100,255,100);
                        cv::circle(path,
                                   cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                                   1,
                                   color,
                                   2
                                    );
                        cv::circle(path,
                                   cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                                   3,
                                   color-cv::Scalar(100),
                                   1
                                    );
                    }
                    else
                    {
                        color = cv::Scalar(255,100,100);
                        cv::circle(path,
                                   cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                                   1,
                                   color,
                                   2
                                    );
                        cv::circle(path,
                                   cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                                   3,
                                   color-cv::Scalar(100),
                                   1
                                    );
                    }
                }
                else
                {
                    color = cv::Scalar(0,255,255);
                    cv::circle(path,
                               cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                               1,
                               color,
                               2
                                );
                    cv::circle(path,
                               cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                               1,
                               color-cv::Scalar(100),
                               2
                                );
                }
                cv::addText(path, 
                            boost::lexical_cast<std::string>(actualFramePose->map_index_), 
                            cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX + 3), ceil(scale*-actualPosisiton[1] + kPathTranslationY - 3)),
                            cv::fontQt("Arial", 7, cv::Scalar(255,255,255))
                           );
            }
        
            oldPosition = actualPosisiton;
        }
        
        // Get the last pose and draw a red empty circle
        actualPosisiton = (*(actualFramePose-1)).pose_T_;
        cv::circle(path,
                   cv::Point(ceil(scale*actualPosisiton[0] + kPathTranslationX), ceil(scale*-actualPosisiton[1] + kPathTranslationY)),
                   3,
                   cv::Scalar(0,0,255),
                   1
                   );
        
    }
    
    void MatchViewer::infoDisplay(cv::Mat& display, const std::string text)
    {
        // reset the display
        display = cv::Scalar(0);
    
        cv::addText(display, text, cv::Point(10,18), font_text_);
    }

    void MatchViewer::drawKeyPoints(cv::Mat& frame, std::vector< cv::KeyPoint > &kpts, std::vector< std::vector< int > >& pointIDXOfCLusters, std::vector<int> matchIdx)
    {
        if (matchIdx.size() <= 0)
        {
            return;
        }
        
        std::vector< std::pair< cv::KeyPoint, int > > kptsIDXpairVector;
        for (std::size_t t = 0; t < kpts.size(); t++)
        {
            kptsIDXpairVector.push_back(std::pair<cv::KeyPoint, int>(kpts[t], -1));
        }
        
        std::cout << "Match size: " << matchIdx.size() << " - ";
        
        for (std::size_t h = 0; h < matchIdx.size(); h++)
        {
            std::cout << " " << matchIdx[h];
            kptsIDXpairVector[matchIdx[h]].second = h;
        }
        std::cout << std::endl;
        
        int thickness = 1;
        cv::Point center;
        cv::Scalar colour;
        int red = 200, blue = 200, green = 255;
        int radius = 80;
        colour = CV_RGB(red, green, blue);
        
        int k = 0;
        for (std::vector< std::pair<cv::KeyPoint, int> >::iterator j = kptsIDXpairVector.begin(); j != kptsIDXpairVector.end(); j++, k++)
        {
            // Draw keypoints
            center = (*j).first.pt;
            
            // Mul for the shift parameter in the circle function
            center.x *= 16;
            center.y *= 16;
            
            cv::circle(frame, center, radius, colour, thickness, CV_AA, 4);
            
            // highlight some cluster descriptors
            if ((*j).second != -1)
            {
                cv::addText(frame, 
                            boost::lexical_cast<std::string>(k) + "-" + boost::lexical_cast<std::string>((*j).second), 
                            (*j).first.pt,
                            cv::fontQt("Arial", 25, cv::Scalar(255,50,50))
                           );
            }
            else
            {
                cv::addText(frame, 
                            boost::lexical_cast<std::string>(k), 
                            (*j).first.pt,
                            cv::fontQt("Arial", 20, cv::Scalar(100,180,255))
                        );
            }
        }
    }

    
} // namespace LCD