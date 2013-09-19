/**
 * \file frameposelist.h
 * \Author: Michele Marostica
 * \brief: This class is the structure that keep track of input frame pose and image path.
 *         It also keep information for each frame if it is in the map and the map index.
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

#ifndef FRAMEPOSELIST_H
#define FRAMEPOSELIST_H

#include <string>
#include <vector>

#include "framepose.h"

namespace LCD {

    typedef typename std::vector<FramePose>::const_iterator const_iterator;
    
    class FramePoseList
    {

    public:
        
        FramePoseList(cv::FileStorage &settings);
        
        virtual ~FramePoseList();
        
        /** Add a FramePose to the list, if it is only a pose: set the framePath to ""; if the frame is not on the map ignore the last parameters
         * @param framePath the filename path of the frame related to the pose
         * @param poseT the translation from the origin of this framepose
         * @param poseR the orientation from the origin of this framepose
         * @param timestamp the timestamp of this framepose
         * @param isOnMap true if the frame has been added to the map
         * @param mapIndex the index of the framepose in the map, -1 if it isn't in the map
         * @param bow the BoW representation of the frame
         * @param kpts the keypoints of the frame
         * @param pointIDXOfCLusters pointIDXOfCLusters[i] contains the indices of the keypoints that has been related to the word i in the vocabulary
         * @param completeDescriptors the matrix with the complete set of descriptors of the related frame
         */
        void add(std::string framePath, cv::Vec3f &poseT, cv::Vec3f &poseR, 
                 unsigned int timestamp, bool isOnMap = false, int mapIndex = -1, 
                 cv::Mat *bow = 0, std::vector<cv::KeyPoint> *kpts = 0, 
                 std::vector< std::vector < int > > *pointIDXOfCLusters = 0,
                 cv::Mat *completeDescriptors = 0
                );
        
        void addDescriptorsToMapFrame(int mapIndex, std::vector<cv::KeyPoint> *kpts, 
                                      std::vector< std::vector<int> > *pointIDXOfCLusters, 
                                      cv::Mat *completeDescriptors, std::vector< std::pair<int, int> > *matchesIndices = 0,
                                      cv::Mat *accorpatedDescriptors = 0 );
        
        const_iterator Begin() const { return list_->begin(); }
        const_iterator End() const { return list_->end(); }
        
        /** Deprecated: return the frame filepath of a specific index
         * @param [in] mapIndex the map index to retive
         * @return the frame filepath of the wanted framepose
         */
        std::string getMapFrameAt(unsigned int mapIndex);
        
        /** Get the framepose at a specific map index
         * @param [in] mapIndex the map index to retive
         * @param [out] frame the returned framepose
         */
        void getMapFramePoseAt(unsigned int mapIndex, FramePose &frame) const;
        
        /** IROS13 Whelan's paper inspired metric
         * @param T the actual translation from origin
         * @param R the actual orientation from origin
         * @return the "distance" computed with the heuristic metric described in the paper
         */
        double distanceFromLastMapFrame(cv::Vec3f T, cv::Vec3f R);
        
        /** IROS13 Whelan's paper inspired descriptor matcher
         * @param queryDescriptors the set of descriptors of a new image
         * @param mapIndexToCompare the index of the frame pose in the map to be compared to the new one
         * @return the number of descriptor matches, computed using FLANN library and a fixed minimum distance threshold
         */
        int descriptorMatcher(cv::Mat* queryDescriptors, int mapIndexToCompare, std::vector< std::pair< int, int > >& matchIndices);
        
    private:
        
        FramePoseList();
        
        std::vector<FramePose>
            *list_;
            
        double descriptorsMatcher_epsilon_;
    };

} // namespace LCD

#endif // FRAMEPOSELIST_H
