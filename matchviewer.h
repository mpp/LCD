/**
 * \file matchviewer.h
 * \Author: Michele Marostica
 * \brief: This class will wrap the visualization code. It will display a window
 *         with the current frame, the most similar one, the path, a plot of the
 *         match output and other informations
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

#ifndef MATCHVIEWER_H
#define MATCHVIEWER_H

#include <opencv2/opencv.hpp>

#include "frameposelist.h"

namespace LCD {
    
    class MatchViewer
    {
    public:
        
        /** Set up the window with static const values of this class
         * TODO: grab these values from a configuration file
         */
        MatchViewer();
        
        /** Update the window mat and return it
         * @param [out] window the updated window to be displaied in a named window
         * @param [in] actualFrame a matrix CV_8UC3 with the actual frame
         * @param [in] similarFrame a matrix CV_8UC3 with the most similar frame
         * @param [in] list the frame-pose list to reconstruct the robot path
         * @param [in] matchArray an array with the match values (will be plot)
         */
//         void update(cv::Mat &window, const cv::Mat &actualFrame, unsigned int actualFrameIndex, const cv::Mat &similarFrame, int similarFrameMapIndex, const FramePoseList &list, std::vector<double> &matchArray, std::vector< std::vector < int > > &pointIDXOfCLusters, std::string rightInfoDisplay);
        void update(cv::Mat& window, int similarFrameMapIndex, bool isMatchAccept,
            const FramePoseList& list, std::vector< cv::DMatch >& matchArray,
            std::vector< std::pair<int,int> > &matchesIndices, std::string leftInfoDisplay, std::string rightInfoDisplay);

    private:
        
        /** Draw the path and return as an image matrix
         * @param [out] path the image matrix of the path
         * @param [in] list the frame-pose list to reconstruct the robot path
         */
        void drawPath(cv::Mat &path, const FramePoseList &list, int similarFrameMapIndex, bool isMatchAccept);
        
        /** Plot the array
         * @param [out] plot the plotted array
         * @param [in] matchArray a row matrix with the results
         */
        void plotArray(cv::Mat &plot, std::vector<cv::of2::IMatch> &matchArray);
        void plotArray(cv::Mat &plot, std::vector<cv::DMatch> &matchArray);
        
        /** Display a string into a display matrix
         * @param [out] display the display matrix
         * @param [in] text the displaied text
         */
        void infoDisplay(cv::Mat &display, const std::string text);
        
        void drawKeyPoints(cv::Mat& frame, std::vector< cv::KeyPoint >& kpts, std::vector< std::vector< int > >& pointIDXOfCLusters, std::vector< int > matchIdx);
        
    private:
        
        cv::Mat
            window_,                    //> The main window
            actual_frame_,              //> actual frame math
            best_match_frame_,          //> best match frame
            match_plot_,                //> the plot of the match result
            path_,                      //> the path
            left_info_display_,         //> an optional informations display
            right_info_display_;        //> an optional informations display
            
        CvFont 
            font_alert_,                  //> the font used into the window
            font_text_;
        
        static const int 
        
            kWindowHeight = 750,        //> the height of the window
            kWindowWidth = 1200,        //> the width of the window
            
            kDisplayHeight = 24,
            kDisplayWidth = 585,
            
            kDisplayTop = 452,
            kLeftDisplayLeft = 10,
            kRightDisplayLeft = 605,
            
            kFrameHeight = 434,         //> the height of a frame into the window
            kFrameWidth = 585,          //> the width of a frame into the window
            
            kActualFrameTop = 10,       //> the top-left position of actual frame
            kActualFrameLeft = 10,
            
            kSimilarFrameTop = 10,      //> the top-left position of similar frame
            kSimilarFrameLeft = 605,
            
            kPathDimension = 256,       //> the dimension of the square for the path
            
            kPathTop = 484,             //> the top left position of the path
            kPathLeft = 934,
            
            kPathScale = 227,           //> 100*the scale of the path
            kPathTranslationX = 70,      //> transition of the (0,0) point of the path
            kPathTranslationY = 170,      //> transition of the (0,0) point of the path
            
            kPlotWidth = 914,           //> the dimensions of the plot
            kPlotHeight = 256,
            
            kPlotTop = 484,             //> the top-left position of the plot
            kPlotLeft = 10;
    };

} // namespace LCD

#endif // MATCHVIEWER_H
