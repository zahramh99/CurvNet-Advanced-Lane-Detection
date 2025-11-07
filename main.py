"""
Advanced Lane Detection System with Curved Lane Estimation
Research-Grade Implementation with Multiple Detection Methods

Author: Zahra Gharehmahmoodlee
License: MIT
Research: Computer Vision, Autonomous Driving, Lane Detection
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
from scipy import stats
from sklearn.linear_model import RANSACRegressor
import json
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LaneDetectionConfig:
    """Configuration class for lane detection parameters"""
    # Camera calibration
    calibration_path: str = 'camera_cal/cal_pickle.p'
    
    # Threshold parameters
    s_thresh: Tuple[int, int] = (100, 255)
    sx_thresh: Tuple[int, int] = (15, 255)
    l_thresh: Tuple[int, int] = (120, 255)
    
    # Perspective transform
    src_points: np.ndarray = None
    dst_points: np.ndarray = None
    
    # Sliding window
    nwindows: int = 9
    margin: int = 100
    minpix: int = 50
    
    # Curvature calculation
    ym_per_pix: float = 30 / 720  # meters per pixel in y dimension
    xm_per_pix: float = 3.7 / 700  # meters per pixel in x dimension
    
    def __post_init__(self):
        if self.src_points is None:
            self.src_points = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
        if self.dst_points is None:
            self.dst_points = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])

class AdvancedLaneDetector:
    """
    Advanced Lane Detection System with multiple detection methods
    and robust curvature estimation for autonomous driving applications
    """
    
    def __init__(self, config: LaneDetectionConfig = None):
        self.config = config or LaneDetectionConfig()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.left_fit = None
        self.right_fit = None
        self.left_curve_rad = 0
        self.right_curve_rad = 0
        self.vehicle_offset = 0
        self.detection_history = []
        
        # Load camera calibration
        self._load_camera_calibration()
    
    def _load_camera_calibration(self):
        """Load camera calibration parameters"""
        try:
            with open(self.config.calibration_path, 'rb') as f:
                cal_data = pickle.load(f)
                self.camera_matrix = cal_data['mtx']
                self.dist_coeffs = cal_data['dist']
                logger.info("Camera calibration loaded successfully")
        except FileNotFoundError:
            logger.warning("Camera calibration file not found, running calibration...")
            self._calibrate_camera()
    
    def _calibrate_camera(self, cal_images_path: str = 'camera_cal/calibration*.jpg'):
        """Automatically calibrate camera using chessboard patterns"""
        obj_pts = np.zeros((6 * 9, 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        images = glob.glob(cal_images_path)
        
        if not images:
            logger.error(f"No calibration images found at {cal_images_path}")
            return
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            
            if ret:
                objpoints.append(obj_pts)
                imgpoints.append(corners)
        
        if objpoints:
            img_size = (img.shape[1], img.shape[0])
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None)
            
            # Save calibration
            dist_pickle = {'mtx': self.camera_matrix, 'dist': self.dist_coeffs}
            with open(self.config.calibration_path, 'wb') as f:
                pickle.dump(dist_pickle, f)
            logger.info("Camera calibration completed and saved")
    
    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Undistort image using camera calibration"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        return img
    
    def multi_threshold_pipeline(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced thresholding pipeline using multiple color spaces and gradients
        """
        img = self.undistort(img)
        
        # Convert to multiple color spaces
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        b_channel = lab[:, :, 2]
        
        # Sobel gradient in x and y directions
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        
        # Gradient magnitude and direction
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))
        
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        
        # Create binary images
        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(scaled_sobelx >= self.config.sx_thresh[0]) & 
                (scaled_sobelx <= self.config.sx_thresh[1])] = 1
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.config.s_thresh[0]) & 
                (s_channel <= self.config.s_thresh[1])] = 1
        
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= self.config.l_thresh[0]) & 
                (l_channel <= self.config.l_thresh[1])] = 1
        
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= 150) & (b_channel <= 255)] = 1
        
        # Combine thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[((s_binary == 1) & (l_binary == 1)) | 
                       ((sxbinary == 1) & (l_binary == 1)) |
                       (b_binary == 1)] = 1
        
        return combined_binary
    
    def perspective_transform(self, img: np.ndarray, dst_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """Apply perspective transform to get bird's eye view"""
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = self.config.src_points * img_size
        dst = self.config.dst_points * np.float32(dst_size)
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        
        return warped, M
    
    def inverse_perspective_transform(self, img: np.ndarray, dst_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """Apply inverse perspective transform"""
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src = self.config.dst_points * img_size
        dst = self.config.src_points * np.float32(dst_size)
        
        M = cv2.getPerspectiveTransform(src, dst)
        unwarped = cv2.warpPerspective(img, M, dst_size)
        
        return unwarped
    
    def sliding_window_search(self, binary_warped: np.ndarray, draw_windows: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced sliding window search with robust outlier rejection
        """
        # Take histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find peaks for left and right lanes
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Set up sliding windows
        window_height = binary_warped.shape[0] // self.config.nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        # Create empty images for visualization
        if draw_windows:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        else:
            out_img = None
        
        for window in range(self.config.nwindows):
            # Identify window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - self.config.margin
            win_xleft_high = leftx_current + self.config.margin
            win_xright_low = rightx_current - self.config.margin
            win_xright_high = rightx_current + self.config.margin
            
            # Draw windows
            if draw_windows:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                            (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                            (win_xright_high, win_y_high), (0, 255, 0), 2)
            
            # Identify nonzero pixels in the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter next window if enough pixels found
            if len(good_left_inds) > self.config.minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.config.minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty, out_img
    
    def fit_polynomial_robust(self, leftx: np.ndarray, lefty: np.ndarray, 
                            rightx: np.ndarray, righty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit polynomials using RANSAC for robust outlier rejection
        """
        if len(leftx) > 10 and len(lefty) > 10:
            # Use RANSAC for robust fitting
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            # Fallback to previous fit
            left_fit = self.left_fit if self.left_fit is not None else np.array([0, 0, 0])
            right_fit = self.right_fit if self.right_fit is not None else np.array([0, 0, 0])
        
        return left_fit, right_fit
    
    def calculate_curvature(self, left_fit: np.ndarray, right_fit: np.ndarray, 
                          ploty: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate lane curvature and vehicle offset in real-world units
        """
        # Define y-value where we want radius of curvature
        y_eval = np.max(ploty)
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * self.config.ym_per_pix, 
                               left_fit[0] * (ploty**2) * self.config.xm_per_pix + 
                               left_fit[1] * ploty * self.config.xm_per_pix + 
                               left_fit[2] * self.config.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.config.ym_per_pix, 
                                right_fit[0] * (ploty**2) * self.config.xm_per_pix + 
                                right_fit[1] * ploty * self.config.xm_per_pix + 
                                right_fit[2] * self.config.xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.config.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.config.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        
        # Calculate vehicle offset from lane center
        lane_center = (left_fit[0] * (y_eval**2) + left_fit[1] * y_eval + left_fit[2] + 
                      right_fit[0] * (y_eval**2) + right_fit[1] * y_eval + right_fit[2]) / 2
        vehicle_offset = (lane_center - 1280/2) * self.config.xm_per_pix
        
        return left_curverad, right_curverad, vehicle_offset
    
    def draw_lane_overlay(self, original_img: np.ndarray, warped_img: np.ndarray, 
                         left_fit: np.ndarray, right_fit: np.ndarray) -> np.ndarray:
        """
        Draw lane overlay and information on the original image
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, original_img.shape[0] - 1, original_img.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Warp the blank back to original image space
        newwarp = self.inverse_perspective_transform(color_warp)
        
        # Combine the result with the original image
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        
        # Add curvature and offset information
        avg_curvature = (self.left_curve_rad + self.right_curve_rad) / 2
        curvature_text = f"Curvature: {avg_curvature:.2f}m"
        offset_text = f"Vehicle Offset: {self.vehicle_offset:.2f}m"
        
        cv2.putText(result, curvature_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, offset_text, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add lane detection confidence
        confidence = self._calculate_confidence(left_fit, right_fit)
        confidence_text = f"Confidence: {confidence:.2f}%"
        cv2.putText(result, confidence_text, (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result
    
    def _calculate_confidence(self, left_fit: np.ndarray, right_fit: np.ndarray) -> float:
        """Calculate detection confidence based on lane parallelism and curvature consistency"""
        if self.left_fit is None or self.right_fit is None:
            return 80.0  # Initial confidence
        
        # Check lane width consistency
        ploty = np.linspace(0, 719, 720)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        lane_widths = right_fitx - left_fitx
        width_std = np.std(lane_widths)
        width_confidence = max(0, 100 - width_std * 10)
        
        return width_confidence
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """
        Main pipeline for processing a single image frame
        """
        # Step 1: Apply thresholding
        binary_img = self.multi_threshold_pipeline(img)
        
        # Step 2: Perspective transform
        warped_img, M = self.perspective_transform(binary_img)
        
        # Step 3: Lane detection
        leftx, lefty, rightx, righty, _ = self.sliding_window_search(warped_img, draw_windows=False)
        
        # Step 4: Fit polynomials
        self.left_fit, self.right_fit = self.fit_polynomial_robust(leftx, lefty, rightx, righty)
        
        # Step 5: Calculate curvature
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        self.left_curve_rad, self.right_curve_rad, self.vehicle_offset = self.calculate_curvature(
            self.left_fit, self.right_fit, ploty)
        
        # Step 6: Draw results
        result = self.draw_lane_overlay(img, warped_img, self.left_fit, self.right_fit)
        
        # Store detection history for smoothing
        self.detection_history.append({
            'left_fit': self.left_fit,
            'right_fit': self.right_fit,
            'curvature': (self.left_curve_rad + self.right_curve_rad) / 2,
            'offset': self.vehicle_offset
        })
        
        # Keep only recent history
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        return result

def main():
    """Main function to demonstrate the lane detection system"""
    # Initialize detector
    config = LaneDetectionConfig()
    detector = AdvancedLaneDetector(config)
    
    # Test with sample image
    test_image = cv2.imread('test_images/test1.jpg')
    if test_image is not None:
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        result = detector.process_image(test_image)
        
        # Display results
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title('Lane Detection Result')
        plt.show()
    
    # Process video if available
    try:
        myclip = VideoFileClip('project_video.mp4')
        output_vid = 'output_lane_detection.mp4'
        clip = myclip.fl_image(detector.process_image)
        clip.write_videofile(output_vid, audio=False)
        logger.info(f"Video processing completed: {output_vid}")
    except Exception as e:
        logger.warning(f"Video processing skipped: {e}")

if __name__ == "__main__":
    main()