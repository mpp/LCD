/**
 * \file frameposelist.cpp
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

#include "frameposelist.h"

#include <limits>

namespace LCD {
    
    FramePoseList::FramePoseList(cv::FileStorage &settings)
    {
        list_ = new std::vector<FramePose>();
        settings["loopDetectorOptions"]["descriptorsMatcherNNDR"] >> descriptorsMatcher_epsilon_;
    }

    FramePoseList::~FramePoseList()
    {
    }

    
    void FramePoseList::add(std::string framePath, cv::Vec3f &poseT, cv::Vec3f &poseR, 
                            unsigned int timestamp, bool isOnMap, int mapIndex, 
                            cv::Mat *bow, std::vector<cv::KeyPoint> *kpts, 
                            std::vector< std::vector < int > > *pointIDXOfCLusters,
                            cv::Mat *completeDescriptors
                           )
    {
        FramePose temp(framePath, poseT, poseR, timestamp, isOnMap, mapIndex);
        
        if (bow != 0)
        {
            temp.bow_ = *bow;
        }
        
        if (kpts != 0)
        {
            temp.kpts_ = *kpts;
        }
        
        if (pointIDXOfCLusters != 0)
        {
            temp.point_IDX_of_clusters_ = *pointIDXOfCLusters;
        }
        
        if (completeDescriptors != 0)
        {
            temp.complete_descriptors_ = * completeDescriptors;
        }
        
        list_->push_back(temp);
    }
    
    void FramePoseList::addDescriptorsToMapFrame(int mapIndex, std::vector< cv::KeyPoint >* kpts, 
                                                 std::vector< std::vector< int > >* pointIDXOfCLusters, 
                                                 cv::Mat* completeDescriptors, std::vector< std::pair<int, int> > *matchesIndices,
                                                 cv::Mat *accorpatedDescriptors )
    {
        for ( std::vector<FramePose>::iterator l = list_->begin(); l != list_->end(); l++ )
        {
            if (l->is_on_map_)
            {
                if (l->map_index_ == mapIndex)
                {
                    if (0 != matchesIndices)
                    {
                        std::vector< std::pair<int, int> >::iterator m_it = matchesIndices->begin();
                        std::vector< cv::KeyPoint >::iterator k_it_base = kpts->begin();
                        std::vector< std::vector<int> >::iterator i_it_base = pointIDXOfCLusters->begin();
                        int actualIndex = 0;
                        
                        std::vector<int>
                            matchIndices;
                            
                        // If these sizes are equals we ignore the descriptor update
                        if (matchIndices.size() == l->complete_descriptors_.rows || completeDescriptors->rows == 0)
                        {
                            return;
                        }
                        
                        // Iterate and swap old kpts with new kpts data
                        while (actualIndex < matchesIndices->size())
                        {
                            int 
                                itemToRemove = m_it->second,
                                newItem = m_it->first;
                            
                            matchIndices.push_back(newItem);
                                
                            l->kpts_.at(itemToRemove) = kpts->at(newItem);
                            l->point_IDX_of_clusters_.at(itemToRemove) = pointIDXOfCLusters->at(newItem);
                            
                            // swap the mat row itemToRemove
                            for (std::size_t col = 0; col < l->complete_descriptors_.cols; col++)
                            {
                                l->complete_descriptors_.at<float>(itemToRemove, col) = completeDescriptors->at<float>(newItem, col);
                            }
                                
                            // Update indices
                            m_it++;
                            actualIndex++;
                        }
                        
                        // Update the descriptors matrix
                        cv::Mat 
                            unmatchedDescriptors(cv::Size(completeDescriptors->cols, completeDescriptors->rows - matchesIndices->size()), 
                                                 CV_32FC1, cv::Scalar(0));
                            
                        // sort the match descriptor's indices
                        std::sort(matchIndices.begin(), matchIndices.end());
                        // index for the matches indices vector
                        int actualMatchIndex = 0;
                        // index for new matrix rows
                        int nextRowToWrite = 0;
                        
                        for (std::size_t actualRow = 0; actualRow < completeDescriptors->rows; actualRow++)
                        {
                            // if actual row is relative to a match: do nothing, update the match index
                            if (matchIndices[actualMatchIndex] == actualRow)
                            {
                                actualMatchIndex++;
                                // Remove elements from vectors
                                kpts->erase((k_it_base+actualRow));
                                k_it_base = kpts->begin();
                                
                                pointIDXOfCLusters->erase((i_it_base+actualRow));
                                i_it_base = pointIDXOfCLusters->begin();
                            }
                            // else copy the descriptor in the new descriptor matrix
                            else
                            {
                                // copy the row
                                for (std::size_t col = 0; col < completeDescriptors->cols; col++)
                                {
                                    unmatchedDescriptors.at<float>(nextRowToWrite, col) = completeDescriptors->at<float>(actualRow, col);
                                }
                                nextRowToWrite++;
                            }
                        }
                        
                        (*completeDescriptors) = unmatchedDescriptors.clone();
                        
                    }
                    
                    std::cout << l->kpts_.size() << "+" << kpts->size() << " = ";
                    l->kpts_.reserve(kpts->size());
                    l->kpts_.insert(l->kpts_.end(), kpts->begin(), kpts->end());
                    std::cout << l->kpts_.size() << std::endl;
                    
                    std::cout << l->point_IDX_of_clusters_.size() << "+" << pointIDXOfCLusters->size() << " = ";
                    l->point_IDX_of_clusters_.reserve(pointIDXOfCLusters->size());
                    l->point_IDX_of_clusters_.insert(l->point_IDX_of_clusters_.end(), pointIDXOfCLusters->begin(), pointIDXOfCLusters->end());
                    std::cout << l->point_IDX_of_clusters_.size() << std::endl;
                    
                    std::cout << l->complete_descriptors_.size() << "+" << completeDescriptors->size() << "=";
                    int baseSize = l->complete_descriptors_.rows;
                    cv::Mat temp = l->complete_descriptors_.clone();
                    l->complete_descriptors_ = cv::Mat(cv::Size(completeDescriptors->cols, completeDescriptors->rows + baseSize), CV_32FC1, cv::Scalar(0));
                    
                    if (completeDescriptors->rows > 0 && completeDescriptors->cols > 0){
                        temp.copyTo(l->complete_descriptors_(cv::Rect(cv::Point2f(0,0), temp.size())));
                        completeDescriptors->copyTo(l->complete_descriptors_(cv::Rect(cv::Point2f(0,temp.rows), completeDescriptors->size())));
                    }
                    std::cout << l->complete_descriptors_.size() << std::endl;
                    
                    if (accorpatedDescriptors != 0)
                    {
                        (*accorpatedDescriptors) = l->complete_descriptors_.clone();
                    }
                    return;
                }
            }
        }
    }
    
    std::string FramePoseList::getMapFrameAt(unsigned int mapIndex)
    {
        for ( std::vector<FramePose>::iterator l = list_->begin(); l != list_->end(); l++ )
        {
            if (l->is_on_map_)
            {
                if (l->map_index_ == mapIndex)
                {
                    return l->frame_path_;
                }
            }
        }
        
        return "";
    }
    
    void FramePoseList::getMapFramePoseAt(unsigned int mapIndex, FramePose& frame) const
    {
        for ( std::vector<FramePose>::iterator l = list_->begin(); l != list_->end(); l++ )
        {
            if (l->is_on_map_)
            {
                if (l->map_index_ == mapIndex)
                {
                    frame = (*l);
                    return;
                }
            }
        }
    }

    double FramePoseList::distanceFromLastMapFrame(cv::Vec3f T, cv::Vec3f R)
    {
        double
            distance = std::numeric_limits< double >::max();
        
        if (list_->size() <= 0)
        {
            return distance;
        }
        
        // iterate to last map frame to get the pose to compare
        cv::Vec3f
            mapT(0,0,0), mapR(0,0,0);
        
        bool
            found = false;
            
        for (std::vector<FramePose>::reverse_iterator l = list_->rbegin(); l != list_->rend(); l++)
        {
//             (*l).print();
            if ((*l).frame_path_ != "")
            {
//                 std::cout << (*l).map_index_ << std::endl;
                mapT = (*l).pose_T_;
                mapR = (*l).pose_R_;
                found = true;
                break;
            }
        }
            
        if (found)
        {
            // Compute the distance
            double 
                translationDistance = 0.0,
                rotationDistance = 0.0;
                
            // Compute L2 norm of the translation
            translationDistance = cv::norm(T, mapT);
            
            cv::Vec3d
                R0 = mapR,
                R1 = R;
            
            // Declare euler' form vectors
            cv::Matx13d
                rDiff(1, 3, CV_64F);
                rDiff(0) = rDiff(1) = rDiff(2) = 0.0;
            
            // Declare rotation maticies
            cv::Matx33d
                q1(3, 3, CV_64F), 
                q2(3, 3, CV_64F), 
                qDiff(3, 3, CV_64F);
            
            // Compute matricies
            cv::Rodrigues(R0,q1);
            
            cv::Rodrigues(R1,q2);
            
            // Compute q' = q1^(-1) * q2
            qDiff = q1.inv() * q2;
            
            // Return to vector representation
            cv::Mat 
                diff;
            cv::Rodrigues(qDiff, diff);
            
            rDiff(0) = diff.at<double>(0,0);
            rDiff(1) = diff.at<double>(0,1);
            rDiff(2) = diff.at<double>(0,2);
                
            // Compute rotation norm
            rotationDistance = cv::norm<double>(rDiff);
            
            distance = translationDistance + rotationDistance;
        }
        return distance;
    }

    int FramePoseList::descriptorMatcher(cv::Mat *queryDescriptors, int mapIndexToCompare, std::vector< std::pair<int, int> > &matchIndices)
    {
        if (mapIndexToCompare < 0)
        {
            return 0;
        }
        
        // Search the related frame pose
        
        FramePose
            *toCompare = 0;
        
        for (std::vector<FramePose>::iterator l = list_->begin(); l != list_->end(); l++)
        {
            if ((*l).is_on_map_)
            {
                if ((*l).map_index_ == mapIndexToCompare)
                {
                    toCompare = &(*l);
                    break;
                }
            }
        }
        
        if (toCompare != 0)
        {
            // Matching descriptor vectors using FLANN matcher
            cv::FlannBasedMatcher 
                matcher;
            std::vector< std::vector< cv::DMatch > >
                matches;
            matcher.knnMatch(*queryDescriptors, toCompare->complete_descriptors_, matches, 2);
            
            int 
                count = 0;
                
            std::vector< std::vector< cv::DMatch> >::iterator 
                j;
                
//             std::cout << std::endl << "Matches:";
            for (j = matches.begin(); j != matches.end(); j++)
            {
                if ((*j).size() >= 2)
                {
                    // Nearest Neighbor Distance Ratio (NNDR) - See section 4.1.1 of MikolajczykPAMI05 paper
                    if ((*j).at(0).distance <= descriptorsMatcher_epsilon_ * (*j).at(1).distance)
                    {
                        count = count + 1;
                        matchIndices.push_back(std::pair<int, int>((*j).at(0).queryIdx, (*j).at(0).trainIdx));
//                         std::cout << " (" << (*j).at(0).queryIdx << " - " << (*j).at(0).trainIdx << ")";
                    }
                }
            }
//             std::cout << std::endl;
            return count;
        }
        else
        {
            return 0;
        }
        
    }

    
} // namespace LCD

