from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self):

        """
        Args:
        max_age: Maximum number of frames to keep track alive without detection
        n_init: Number of consecutive detections before track is confirmed
        nn_budget: Maximum size of appearance gallery
        """
        self.tracker = DeepSort(
            max_age=100,           # Keep tracks
            n_init=5,              # confirmations before track is confirmed
            nms_max_overlap=0.7,  
            max_cosine_distance=0.6,
            nn_budget=50,          # Appearance gallery
            embedder="mobilenet",
            half=True,
            bgr=True
        )

    def update(self, detections, frame):
        """
        Args:
            detections: List of detections from YOLO detector
            frame: Current frame for appearance feature extraction
            
        Returns:
            tracks: List of confirmed tracks with IDs
        """

        try:
            # Convert detections to DeepSORT format
            detection_list = []
            for detection in detections:
                bbox, confidence, class_id = detection
                detection_list.append((bbox, confidence, class_id))
            
            # Update tracker
            tracks = self.tracker.update_tracks(detection_list, frame=frame)
            
            # Extract confirmed tracks
            confirmed_tracks = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()  # Get bounding box in (left, top, right, bottom) format
                
                confirmed_tracks.append({
                    'id': track_id,
                    'bbox': ltrb,
                    'class_id': track.det_class if hasattr(track, 'det_class') else 0
                })
                
            return confirmed_tracks
    
        except Exception as e:
            print(f"Error in DeepSORTTracker: {e}")
            import traceback
            traceback.print_exc()
            return []