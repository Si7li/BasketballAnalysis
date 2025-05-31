from utils import read_video, save_video
from trackers import PlayerTracker
def main():
    
    #Read video frames from a file
    video_frames = read_video("input_videos/video_1.mp4")
    
    #Initialize the player tracker with the model path
    player_tracker = PlayerTracker("models/player_detector.pt")

    #Run Trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl")

    print("Player Tracks:", player_tracks)
    #Save video frames to a new file
    save_video(video_frames, "output_videos/output_video.avi")

if  __name__ == "__main__":
    main()