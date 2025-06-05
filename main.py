from utils import read_video, save_video
from trackers import PlayerTracker,BallTracker
from drawer import PlayerTracksDrawer, BallTracksDrawer
from team_assigner import TeamAssigner

def main():
    
    #Read video frames from a file
    video_frames = read_video("input_videos/video_1.mp4")
    
    #Initialize the player tracker with the model path
    player_tracker = PlayerTracker("models/player_detector.pt")

    #Run Trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl")
    
    ball_tracker = BallTracker("models/ball_detector.pt")
    #Run Trackers
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="stubs/ball_track_stubs.pkl")

    #Remove wrong ball detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    #Interpolate ball tracks
    ball_tracks = ball_tracker.interp_ball_positions(ball_tracks)

    #Assign teams to players
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, 
                                                                 player_tracks,
                                                                 read_from_stub=True,
                                                                 stub_path="stubs/player_assignment_stubs.pkl")
    

    #Draw Output
    #Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    #Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks) 

    #Save video frames to a new file
    save_video(output_video_frames, "output_videos/output_video.avi")

if  __name__ == "__main__":
    main()