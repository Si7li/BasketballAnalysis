from utils import read_video, save_video
from trackers import PlayerTracker,BallTracker
from drawer import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer, CourtKeypointDrawer, TacticalViewDrawer, SpeedAndDistanceDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator

def main():
    
    #Read video frames from a file
    video_frames = read_video("input_videos/video_1.mp4")
    
    #Initialize the player tracker with the model path
    player_tracker = PlayerTracker("models/player_detector.pt")

    #Init Court Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector("models/court_keypoint_detector.pt")

    #Run Trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl")
    
    ball_tracker = BallTracker("models/ball_detector.pt")
    #Run Trackers
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="stubs/ball_track_stubs.pkl")

    #Get Court Keypoints
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                  read_from_stub=True,
                                                                  stub_path="stubs/court_keybpoints_stubs.pkl"
                                                                  )
    
    
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
    
    #Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    # Detect Passes and Interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquition, player_assignment)

    #Tactical view
    tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    court_keypoints = tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints,player_tracks)

    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    #Draw Output
    #Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    passes_and_interceptions_drawer = PassInterceptionDrawer()
    court_keypoints_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    #Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                    player_tracks, 
                                                    player_assignment,
                                                    ball_aquition
                                                    )
    output_video_frames = ball_tracks_drawer.draw(output_video_frames,
                                                  ball_tracks) 

    #Draw team ball controll
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_aquition)

    # Draw Passes and Interceptions
    output_video_frames = passes_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions)
    
    # Draw Keypoints
    output_video_frames = court_keypoints_drawer.draw(output_video_frames,
                                                      court_keypoints)
    # Speed and Distance Drawer
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                         player_tracks,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )

    # Tactical View
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_aquition)

    #Save video frames to a new file
    save_video(output_video_frames, "output_videos/output_video.avi")

if  __name__ == "__main__":
    main()