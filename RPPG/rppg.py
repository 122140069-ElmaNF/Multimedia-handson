"""
Real-time Remote Photoplethysmography (rPPG) Heart Rate Detection
Sistem Teknologi Multimedia - Institut Teknologi Sumatera

Features:
- Real-time face detection using MediaPipe
- POS (Plane-Orthogonal-to-Skin) algorithm for robust signal extraction
- Adaptive bandpass filtering
- Motion artifact rejection
- Real-time signal visualization (side-by-side layout)
- ESC key to stop

Controls:
- ESC: Stop and exit
- R: Reset buffers
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import time


class RPPGDetector:
    def __init__(self, window_size=300, fps=30, min_bpm=40, max_bpm=180, mirror=True):
        """
        Initialize rPPG Detector
        
        Args:
            window_size: Number of frames for signal buffer (default: 300 = 10 seconds at 30fps)
            fps: Expected camera frame rate
            min_bpm: Minimum heart rate to detect
            max_bpm: Maximum heart rate to detect
        """
        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Parameters
        self.window_size = window_size
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.min_freq = min_bpm / 60.0  # Convert to Hz
        self.max_freq = max_bpm / 60.0
        
        # Signal buffers - using deque for efficient FIFO
        self.rgb_buffer = deque(maxlen=window_size)  # Store RGB values for POS
        self.signal_buffer = deque(maxlen=window_size)  # Store processed signal
        self.bpm_buffer = deque(maxlen=10)  # Smooth BPM over last 10 estimates
        
        # State variables
        self.current_bpm = 0
        self.confidence = 0
        self.frame_count = 0
        self.last_bpm = 0
        
        # Butterworth filter parameters
        self.filter_order = 6
        self.lowcut = self.min_freq
        self.highcut = self.max_freq
        
        # ROI landmarks sets (MediaPipe face mesh indices)
        # Forehead region (default)
        self.forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # Cheek (left / right) approximations — useful alternative ROI
        self.cheek_left_indices = [234, 93, 132, 58, 172, 136, 150, 149]
        self.cheek_right_indices = [454, 323, 361, 288, 397, 365, 379, 378]

        # ROI selection mapping and current choice (no 'Eyes' ROI)
        self.roi_options = {
            'Forehead': self.forehead_indices,
            'Cheek-Left': self.cheek_left_indices,
            'Cheek-Right': self.cheek_right_indices
        }
        self.current_roi_name = 'Forehead'
        self.current_roi_indices = self.roi_options[self.current_roi_name]
        # Mirror preview control. Default False -> not mirrored.
        # Some webcams show a mirrored preview; flip horizontally when mirror=True
        self.mirror = mirror
        
    def cpu_POS(self, signal_input, fps):
        """
        POS method on CPU using Numpy.
        Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). 
        Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
        
        Args:
            signal_input: RGB signal array of shape [frames, 3]
            fps: Frame rate
        """
        eps = 10**-9
        
        # Reshape input: [frames, 3] -> [1, 3, frames] for algorithm
        X = signal_input.T  # [3, frames]
        X = np.expand_dims(X, axis=0)  # [1, 3, frames]
        
        e, c, f = X.shape  # e = #estimators (1), c = 3 rgb channels, f = #frames
        w = int(1.6 * fps)  # window length (recommended: 1.6 seconds)
        
        if f < w:
            return None
        
        # Stack e times fixed mat P
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)
        
        # Initialize (1)
        H = np.zeros((e, f))
        
        for n in np.arange(w, f):
            # Start index of sliding window (4)
            m = n - w + 1
            
            # Temporal normalization (5)
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2) + eps)
            M = np.expand_dims(M, axis=2)
            Cn = np.multiply(M, Cn)
            
            # Projection (6)
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)
            
            # Tuning (7)
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            
            # Overlap-adding (8)
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
        
        return H[0]  # Return signal for first (and only) estimator
        
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """Apply Butterworth bandpass filter"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
    
    def extract_roi_mask(self, frame, landmarks, indices):
        """Extract ROI mask from facial landmarks"""
        h, w = frame.shape[:2]
        points = []
        
        for idx in indices:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        
        return mask
    
    def extract_rgb_signal(self, frame, mask):
        """Extract RGB spatial average from ROI"""
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        roi_pixels = masked_frame[mask > 0]
        
        if len(roi_pixels) == 0:
            return None
        
        # Spatial averaging - BGR format, so reverse to RGB
        mean_bgr = np.mean(roi_pixels, axis=0)
        mean_rgb = mean_bgr[::-1]  # Reverse BGR to RGB
        
        return mean_rgb
    
    def estimate_bpm_fft(self, signal_data, fps):
        """Estimate BPM using FFT"""
        n = len(signal_data)
        fft_values = fft(signal_data)
        fft_freqs = fftfreq(n, 1.0 / fps)
        
        # Take only positive frequencies
        positive_freqs = fft_freqs[:n // 2]
        positive_fft = np.abs(fft_values[:n // 2])
        
        # Find frequencies within valid range
        valid_idx = np.where((positive_freqs >= self.min_freq) & 
                            (positive_freqs <= self.max_freq))[0]
        
        if len(valid_idx) == 0:
            return 0, 0
        
        valid_freqs = positive_freqs[valid_idx]
        valid_fft = positive_fft[valid_idx]
        
        # Find peak frequency
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_fft[peak_idx]
        
        # Calculate BPM
        bpm = peak_freq * 60.0
        
        # Calculate confidence based on peak prominence
        confidence = peak_power / (np.mean(valid_fft) + 1e-8)
        confidence = min(confidence / 10.0, 1.0)
        
        return bpm, confidence
    
    def smooth_bpm(self, bpm):
        """Apply median filtering to BPM estimates"""
        self.bpm_buffer.append(bpm)
        return np.median(list(self.bpm_buffer))
    
    def draw_signal_plot(self, width=400, height=600):
        """Draw real-time signal plot on a canvas"""
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Title
        cv2.putText(canvas, "Heart Rate Signal", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mirror and ROI indicators (top-right)
        mirror_text = f"Mirror: {'ON' if getattr(self, 'mirror', False) else 'OFF'}"
        roi_text = f"ROI: {getattr(self, 'current_roi_name', 'Forehead')}"
        # Draw right-aligned small text
        cv2.putText(canvas, mirror_text, (width - 10 - 180, 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, roi_text, (width - 10 - 180, 36), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        if len(self.signal_buffer) < 30:
            # Show loading
            progress = len(self.signal_buffer)
            text = f"Collecting data... {progress}/{self.window_size}"
            cv2.putText(canvas, text, (10, height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return canvas
        
        # Get signal data
        signal_data = np.array(list(self.signal_buffer))
        
        # Normalize signal for display
        if len(signal_data) > 0:
            signal_normalized = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-8)
        else:
            signal_normalized = signal_data
        
        # Dynamic plot area dimensions to avoid overlapping other UI elements
        title_height = 40
        plot_y_start = title_height
        plot_height = int(height * 0.55)
        plot_width = width - 40
        plot_x_start = 20
        
        # Draw grid (4 horizontal lines + border)
        for i in range(5):
            y = plot_y_start + i * (plot_height // 4)
            cv2.line(canvas, (plot_x_start, y), (plot_x_start + plot_width, y), (60, 60, 60), 1)
        
        # Draw signal
        num_points = len(signal_normalized)
        if num_points > 1:
            points = []
            for i, value in enumerate(signal_normalized):
                x = int(plot_x_start + (i / num_points) * plot_width)
                y = int(plot_y_start + plot_height - value * plot_height)
                points.append((x, y))
            
            # Draw line
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], (0, 255, 0), 2)
        
        # Draw BPM info below the plot
        bpm_y = plot_y_start + plot_height + 8
        cv2.putText(canvas, f"Heart Rate", (10, bpm_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # BPM value with color based on confidence
        color = (0, 255, 0) if self.confidence > 0.5 else (0, 165, 255)
        cv2.putText(canvas, f"{self.current_bpm:.1f} BPM", (10, bpm_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Confidence bar
        bar_y = bpm_y + 60
        cv2.putText(canvas, "Confidence", (10, bar_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        bar_width = int((width - 40) * self.confidence)
        cv2.rectangle(canvas, (20, bar_y + 10), (20 + bar_width, bar_y + 30), color, -1)
        cv2.rectangle(canvas, (20, bar_y + 10), (width - 20, bar_y + 30), (255, 255, 255), 2)
        
        # Status text
        status_y = bar_y + 60
        if self.confidence > 0.7:
            status = "Status: EXCELLENT"
            status_color = (0, 255, 0)
        elif self.confidence > 0.4:
            status = "Status: GOOD"
            status_color = (0, 255, 255)
        else:
            status = "Status: POOR - Stay still"
            status_color = (0, 165, 255)
        
        cv2.putText(canvas, status, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        instructions = [
            "",
            "Tips:",
            "- Stay still",
            "- Good lighting",
            "- Face camera"
        ]
        
        # Instructions placed at the bottom, below BPM and confidence
        inst_y = bar_y + 60

        # Horizontal controls line (single-line) to avoid vertical stacking
        controls_line = "Controls: ESC | R | M | ROI: 1-Forehead 2-Cheek-L 3-Cheek-R"
        cv2.putText(canvas, controls_line, (10, inst_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        for i, line in enumerate(instructions):
            cv2.putText(canvas, line, (10, inst_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        return canvas
    
    def process_frame(self, frame):
        """Process single frame for rPPG detection"""
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Detect face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, "No face detected"
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract ROI mask (use current selection)
        mask = self.extract_roi_mask(frame, landmarks, self.current_roi_indices)
        
        # Extract RGB signal
        rgb_mean = self.extract_rgb_signal(frame, mask)
        
        if rgb_mean is None:
            return frame, "ROI extraction failed"
        
        # Store RGB values
        self.rgb_buffer.append(rgb_mean)
        
        # Draw ROI on frame
        frame_with_roi = frame.copy()
        overlay = frame_with_roi.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4
        cv2.addWeighted(overlay, 0.5, frame_with_roi, 0.5, 0, frame_with_roi)
        
        # Draw some landmarks
        # Draw a subset of landmarks for the selected ROI
        for idx in self.current_roi_indices[::max(1, len(self.current_roi_indices)//12)]:
            landmark = landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame_with_roi, (x, y), 2, (0, 255, 0), -1)
        
        # Need sufficient data for POS processing
        if len(self.rgb_buffer) < 60:
            status = f"Initializing... {len(self.rgb_buffer)}/{self.window_size}"
            self.draw_simple_info(frame_with_roi, status)
            return frame_with_roi, status
        
        # Process signal every 30 frames (1 second at 30fps)
        if self.frame_count % 30 == 0:
            try:
                # Convert buffer to array [frames, 3]
                rgb_array = np.array(list(self.rgb_buffer))
                
                # Apply POS algorithm
                pos_signal = self.cpu_POS(rgb_array, self.fps)
                
                if pos_signal is None:
                    status = "Processing..."
                    self.draw_simple_info(frame_with_roi, status)
                    return frame_with_roi, status
                
                # Take the last part of signal for filtering
                signal_to_filter = pos_signal[-len(self.rgb_buffer):]
                
                # Apply bandpass filter
                filtered = self.butter_bandpass_filter(
                    signal_to_filter, self.lowcut, self.highcut, self.fps, self.filter_order
                )
                
                # Store filtered signal for visualization
                self.signal_buffer.clear()
                for val in filtered:
                    self.signal_buffer.append(val)
                
                # Estimate BPM using FFT
                bpm, confidence = self.estimate_bpm_fft(filtered, self.fps)
                
                # Smooth BPM
                if confidence > 0.3:
                    self.current_bpm = self.smooth_bpm(bpm)
                    self.confidence = confidence
                
            except Exception as e:
                print(f"Processing error: {e}")
                status = "Processing error"
                self.draw_simple_info(frame_with_roi, status)
                return frame_with_roi, status
        
        # Draw simple BPM on video frame
        status = f"BPM: {self.current_bpm:.1f}"
        self.draw_simple_info(frame_with_roi, status)
        
        return frame_with_roi, status
    
    def draw_simple_info(self, frame, status):
        """Draw minimal info on video frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background (small box so it doesn't cover the whole frame)
        overlay = frame.copy()
        box_right = min(w - 10, 300)
        cv2.rectangle(overlay, (10, 10), (box_right, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text + Mirror and ROI info
        color = (0, 255, 0) if self.confidence > 0.5 else (0, 165, 255)
        cv2.putText(frame, status, (20, 36), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Mirror indicator
        mirror_text = f"Mirror: {'ON' if self.mirror else 'OFF'}"
        cv2.putText(frame, mirror_text, (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ROI display
        cv2.putText(frame, f"ROI: {self.current_roi_name}", (20 + 140, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def run(self):
        """Main loop for real-time rPPG detection"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 60)
        print("Real-time rPPG Heart Rate Detection")
        print("=" * 60)
        print("\nControls:")
        print("  ESC - Stop and exit")
        print("  R   - Reset buffers")
        print("  M   - Toggle mirror (preview flip)")
        print("  1   - ROI: Forehead (default)")
        print("  2   - ROI: Cheek-Left")
        print("  3   - ROI: Cheek-Right")
        print("\nInstructions:")
        print("  1. Position your face in the frame")
        print("  2. Stay still and maintain good lighting")
        print("  3. Wait 5-10 seconds for stable reading")
        print("\nStarting camera...")
        print("=" * 60)
        
        window_name = 'rPPG Heart Rate Detection'
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror frame if enabled
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Process frame
            processed_frame, status = self.process_frame(frame)
            
            # Create signal plot
            signal_plot = self.draw_signal_plot(width=400, height=480)
            
            # Combine video and signal plot side by side
            combined = np.hstack([processed_frame, signal_plot])
            
            # Display
            cv2.imshow(window_name, combined)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            # ESC key to exit (key code 27)
            if key == 27:
                print("\n[ESC] Stopping detection...")
                break
            # R key to reset
            elif key == ord('r') or key == ord('R'):
                self.rgb_buffer.clear()
                self.signal_buffer.clear()
                self.bpm_buffer.clear()
                self.current_bpm = 0
                self.confidence = 0
                self.frame_count = 0
                print("[R] Buffers reset")
            # M key to toggle mirror
            elif key == ord('m') or key == ord('M'):
                self.mirror = not self.mirror
                print(f"[M] Mirror set to: {self.mirror}")
            # ROI selection keys (1..4)
            elif key == ord('1'):
                self.current_roi_name = 'Forehead'
                self.current_roi_indices = self.roi_options[self.current_roi_name]
                print('[1] ROI set to Forehead')
            elif key == ord('2'):
                self.current_roi_name = 'Cheek-Left'
                self.current_roi_indices = self.roi_options[self.current_roi_name]
                print('[2] ROI set to Cheek-Left')
            elif key == ord('3'):
                self.current_roi_name = 'Cheek-Right'
                self.current_roi_indices = self.roi_options[self.current_roi_name]
                print('[3] ROI set to Cheek-Right')
            # (key 4 removed - only 1..3 available)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("Detection stopped. Thank you!")
        print("=" * 60)


def main():
    """Main entry point"""
    # Initialize detector with optimized parameters
    detector = RPPGDetector(
        window_size=300,    # 10 seconds buffer at 30fps
        fps=30,             # Expected frame rate
        min_bpm=40,         # Minimum heart rate
        max_bpm=180         # Maximum heart rate
    )
    # Camera preview is mirrored by default
    detector.mirror = True
    
    # Run detection
    detector.run()


if __name__ == "__main__":
    main()