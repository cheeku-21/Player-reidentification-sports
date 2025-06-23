import cv2
import numpy as np
import time
from YOLOv11_Detector import YOLOv11Detector
from tracker import Tracker

class AdvancedFootballTracker:
    def __init__(self, model_path, video_path=None, output_path=None):

        """
        Args:
            model_path: Path to fine-tuned YOLOv11 model
            video_path: Path to input video
            output_path: Path to save output video
        """
        self.detector = YOLOv11Detector(model_path, conf_threshold=0.8)
        self.tracker = Tracker()
        
        # Re-identification components
        self.lost_tracks = {}  # Store recently lost tracks
        self.track_history = {}  # Store track movement history
        self.appearance_gallery = {}  # Store appearance features
        self.max_lost_time = 60  # Frames to keep lost tracks
        
        self.video_path = video_path
        self.output_path = output_path
        self.cap = None
        self.writer = None

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()

    def initialize_video(self):
        """Initialize video capture and writer"""

        if self.video_path is None:
            self.cap = cv2.VideoCapture(0)  # Webcam
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video source: {self.video_path}")
            
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path specified
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
        return fps, width, height
    
    def draw_tracks(self, frame, tracks):
        """Draw tracking results on frame"""

        for track in tracks:
            track_id = track['id']
            x1, y1, x2, y2 = map(int, track['bbox'])
            
            # Choose color based on track ID for consistency
            color = self.get_track_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"Player {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        return frame
    
    def get_track_color(self, track_id):
        """Generate consistent color for each track ID"""

        # Simple color generation based on ID
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 0, 0)
        ]
        
        # Ensuring track_id is an integer before using modulo
        try:
            track_id_int = int(track_id)
            return colors[track_id_int % len(colors)]
        except (ValueError, TypeError):
            # Return default color if track_id is invalid
            return colors[0]  # Default to red
        
    def update_track_history(self, tracks, frame_count):
        """Update movement history for each track"""

        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame': frame_count,
                'center': (center_x, center_y),
                'bbox': bbox
            })
            
            # Keep only recent history (last 30 frames)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
    
    def predict_reappearance_location(self, track_id):
        """Predict where a lost track might reappear"""

        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return None
            
        history = self.track_history[track_id][-3:]  # Last 3 positions
        
        # Calculate velocity
        dx = history[-1]['center'][0] - history[0]['center'][0]
        dy = history[-1]['center'][1] - history[0]['center'][1]
        
        # Predict next position
        predicted_x = history[-1]['center'][0] + dx
        predicted_y = history[-1]['center'][1] + dy
        
        return (predicted_x, predicted_y)
    
    def attempt_reidentification(self, new_tracks, frame_count):
        """Trying to re-identify lost player"""

        if not self.lost_tracks:
            return new_tracks
        
        # Check for potential re-identifications
        tracks_to_remove = []
        
        for lost_id, lost_info in self.lost_tracks.items():
            if frame_count - lost_info['lost_frame'] > self.max_lost_time:
                tracks_to_remove.append(lost_id)
                continue
            
            # Get predicted location
            predicted_pos = self.predict_reappearance_location(lost_id)
            if predicted_pos is None:
                continue
            
            # Find closest new track
            min_distance = float('inf')
            best_match_idx = -1
            
            for idx, track in enumerate(new_tracks):
                if track['id'] in [t['id'] for t in new_tracks[:idx]]:  # Skip if already processed
                    continue
                
                bbox = track['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Calculate distance to predicted position
                distance = ((center_x - predicted_pos[0])**2 + (center_y - predicted_pos[1])**2)**0.5
                
                if distance < min_distance and distance < 100:  # Within 100 pixels
                    min_distance = distance
                    best_match_idx = idx
            
            # Re-assign ID if good match found
            if best_match_idx >= 0:
                new_tracks[best_match_idx]['id'] = lost_id
                tracks_to_remove.append(lost_id)
                print(f"Re-identified player {lost_id} after {frame_count - lost_info['lost_frame']} frames")
        
        # Clean up old lost tracks
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
        
        return new_tracks
    
    def update_lost_tracks(self, current_track_ids, frame_count):
        """Update list of lost tracks"""

        # Find tracks that disappeared
        previous_ids = set(self.track_history.keys())
        current_ids = set(current_track_ids)
        
        newly_lost = previous_ids - current_ids
        
        for lost_id in newly_lost:
            if lost_id not in self.lost_tracks:
                self.lost_tracks[lost_id] = {
                    'lost_frame': frame_count,
                    'last_position': self.track_history[lost_id][-1] if lost_id in self.track_history else None
                }
                print(f"Player {lost_id} lost at frame {frame_count}")
    
    def run_with_reidentification(self):
        """Enhanced tracking with re-identification"""
        try:
            fps, width, height = self.initialize_video()
            print(f"Video initialized: {width}x{height} @ {fps} FPS")
            
            frame_count = 0
            current_fps = 0.0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Perform detection
                detections = self.detector.detect(frame)
                
                # Update tracker
                tracks = self.tracker.update(detections, frame)
                
                # Apply re-identification
                tracks = self.attempt_reidentification(tracks, frame_count)
                
                # Update tracking history
                self.update_track_history(tracks, frame_count)
                
                # Update lost tracks
                current_track_ids = [track['id'] for track in tracks]
                self.update_lost_tracks(current_track_ids, frame_count)
                
                # Draw results
                frame = self.draw_tracks(frame, tracks)
                
                # Display info
                info_text = f"Frame: {frame_count} | Active: {len(tracks)} | Lost: {len(self.lost_tracks)}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show unique IDs used so far
                all_ids = set(current_track_ids + list(self.lost_tracks.keys()))
                max_id = max(all_ids) if all_ids else 0
                id_info = f"Max ID: {max_id} | Unique players: {len(all_ids)}"
                cv2.putText(frame, id_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Enhanced Tracking', frame)  # For local
                
                if self.writer:
                    self.writer.write(frame)
                    
        except Exception as e:
            print(f"Error during tracking: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

enhanced_tracker = AdvancedFootballTracker(
    model_path="C:/Users/Sudha/Downloads/best.pt",
    video_path="C:/Users/Sudha/Downloads/15sec_input_720p.mp4",
    output_path="final_enhanced_tracking_output.mp4"
)

# Run with re-identification
enhanced_tracker.run_with_reidentification()
