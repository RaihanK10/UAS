import olympe
import os
import requests
from math import radians
from time import sleep
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveTo, moveBy
from olympe.messages.ardrone3.PilotingState import moveToChanged, FlyingStateChanged, PositionChanged
from olympe.enums.ardrone3.Piloting import MoveTo_Orientation_mode
from olympe.messages.camera import set_camera_mode, set_photo_mode, take_photo, photo_progress
from olympe.messages import gimbal
from vision import find_closest_spot
from enum import Enum

##################################
# Some default variables
##################################
# IPs that are where the SDK sends controls
IP_PHYSICAL = "192.168.42.1"  # The connection IP for the PHYSICAL drone device
IP_SIM = "10.202.0.1"  # The connection IP for the simulator drone
DOWNLOAD_DIR = "./"  # directory for downloaded images
FILE_NAME = "capture.jpg"  # file name for a downloaded image
PARKING_SPOT_HOVER_HEIGHT = 0.25 # 25 centimeteres
INCHES_PER_PIXEL = (10 / 3600) * 12 # 10ft / 3600 pixels, represents inches per pixel ratio
METERS_PER_INCH = 0.0254 # 1 inch is 0.0254 meters
DEGREES_PER_METER = 1 / 111_139 # 1 degree is 111 kilometers

class MODE(Enum):
    SIM = IP_SIM
    PHYSICAL = IP_PHYSICAL

# Move the drone to a position and pause the code until it arrives.
# If this fails the program will attempt a landing and shut down
def __move_drone(drone, lat, long, altitude, mode=MoveTo_Orientation_mode.NONE):
    drone(
        moveTo(lat, long, 1.8, mode, 0)
        # Verify how our commands ended up, if not true the control will cease
        >> moveToChanged(latitude=lat, longitude=long, altitude=1.8, orientation_mode=mode, status='DONE',
                         _policy='wait')).wait()    


# Flies the drone to 180cm, a good height for observing the model lot
def __travel_to_observation_height(drone):
    # Initializes the start-up sequence and settings
    # Flies the drone to about 1.1 meters, 3ft 7 in
    assert drone(
        TakeOff()
        >> FlyingStateChanged(state="hovering", _timeout=5)
    ).wait().success()

    lat, long, mode = __getCords(drone)

    # After initial take off, we need to go a little higher
    __move_drone(drone, lat, long, 1.8)


# Positions the camera and the drone such that it can capture the parking lot.
# An image is taken and saved internally on the drone
def __capture_parking_lot(drone):
    __set_camera_modes(drone)
    yaw_moved = __position_camera(drone, 0, -45, -15)

    # Set up  waiter to track the progress of the photo process. handled asynchronously
    photo_status = drone(photo_progress(result="photo_saved", _policy="wait"))
    drone(
        take_photo(0)  # Take a photo using the gimbal camera instead of vertical camera
    ).wait()
    photo_status.wait()
    return yaw_moved, photo_status


# Tell the drone to use configurations for camera
def __set_camera_modes(drone):
    # Set to photo mode; in video mode photography API does not function
    drone(
        set_camera_mode(cam_id=0, value="photo").wait()
    )

    # Pass in photo settings and wait for the drone to confirm the settings update
    assert drone(
        # In Olympe logs, available mode configurations are printed. These are the only configurations allowed.
        set_photo_mode(
            cam_id=0,
            mode="single",
            format="rectilinear",
            file_format="jpeg",
            burst="burst_14_over_1s",
            bracketing="preset_1ev",
            capture_interval=0.0,
        )
    ).wait().success()

    # Repeat setting photo camera to confirm we did not switch back to video mode (rare glitch)
    drone(
        set_camera_mode(cam_id=0, value="photo").wait()
    )


# Move the gimbal to pitch downwards
# Rotate the drone slightly to view left side of parking lot
def __position_camera(drone, roll, pitch, yaw):
    drone(
        gimbal.set_target(
            gimbal_id=0,
            control_mode="position",  # position or velocity
            yaw_frame_of_reference="relative",
            yaw=0.0,  # ANAFI 4k does not have a yaw gimble.
            pitch_frame_of_reference="absolute",
            pitch=pitch,
            roll_frame_of_reference="relative",
            roll=roll
        )
    ).wait()

    # We do not have a yaw gimbal, we need to manually rotate the drone
    return drone(
        moveBy(dX=0, dY=0, dZ=0, dPsi=radians(yaw))  # dPsi takes in radians to move heading (yaw)
    ).wait().success()


# Hit the media endpoint to download the most recent image
def __download_image(current_ip):
    # Query the drone to get a collection of all the onboard files
    discover_url = "http://" + current_ip + "/api/v1/media/medias"
    media = requests.request("GET", discover_url)  # Fetch a list of all the files
    most_recent_photo = media.json()[-1]["resources"][0][
        "url"]  # Slice out the most recent resource, and get the associated 'url' with it

    # Download manually using curl http://192.168.42.1/data/media/100000490049.JPG --output out.jpg
    # where 100000490049 is a resource_id retrieved from the media call 
    fetch_url = "http://" + current_ip + most_recent_photo
    print("Sending request: ", fetch_url)
    requested_file = requests.request("GET", fetch_url)

    # If we requested a real file
    if requested_file.status_code == 200:
        with open(DOWNLOAD_DIR + FILE_NAME, 'wb') as writeable_file:  # download it into something usable
            writeable_file.write(requested_file.content)
    else:
        assert False, "failed to download image"
    del requested_file  # cleanup the leftovers from the request


# Get the current Latitude and Longitude of the drone, used for identifying relative movements
def __getCords(drone):
    lat = drone.get_state(PositionChanged)["latitude"]
    long = drone.get_state(PositionChanged)["longitude"]
    mode = MoveTo_Orientation_mode.NONE
    return lat, long, mode

# Turns our pixel distances into feet/in and then into GPS fractions
# In our model, 3600 pixels = 10 feet, or 1 pixel = 1/30 inches
def __to_gps(pixel_dist):
    degrees = pixel_dist * INCHES_PER_PIXEL * METERS_PER_INCH * DEGREES_PER_METER
    return degrees


# Fl
def __park(x, y, drone):
    current_lat, current_long, mode = __getCords(drone)
    delta_lat = __to_gps(x)
    delta_long = __to_gps(y)

    new_lat = current_lat + delta_lat
    new_long = current_long + delta_long

    __move_drone(drone, new_lat, new_long, PARKING_SPOT_HOVER_HEIGHT)


# Allows the drone to be landed from a remote connection without entering parking loop
def force_land(mode=MODE.PHYSICAL):
    # Connect to the drone
    drone = olympe.Drone(mode.value)
    drone.connect()
    assert drone(Landing()).wait().success()
    drone.disconnect()

# Serial function for finding a parking spot
def take_me_to_my_parking_spot(mode=MODE.PHYSICAL):

    # Connect to the drone
    drone = olympe.Drone(mode.value)
    drone.connect()

    try:
        # Fly to about 1.8 meters
        __travel_to_observation_height(drone)

        # Save our launch position
        home_lat, home_long, mode = __getCords(drone)
        
        # Only the Physical drone has functioning camera, if we're using that device proceede
        if(mode == MODE.PHYSICAL):

            # Move the gimbal for pitch and roll, we must physical turn the device for yaw.
            # Keep track of how much yaw we moved so we can reset our heading later.
            # Additionally track the photo functions progress 
            yaw_moved, photo_progress = __capture_parking_lot(drone)
            
            # We will wait for the image to be available in ${#link:__capture_parking_lot} for 5 seconds. Afterwards we can check if it finished successfully 
            if not photo_progress:
                assert False, "take_photo timeout"
            
            # If we've made it this far, the drone has an image stored for this trip, download onto the remote host (this machine)
            __download_image(mode.value)

            # Find the position on the image of the optimum parking spot
            x, y, distance = find_closest_spot(DOWNLOAD_DIR + FILE_NAME)

        # Since the simulator cannot use the camera, we'll use default values from one of our training sets
        else:
            yaw_moved = 0
            x, y, distance = 1936.1716, 1787.3828, 786.4168
        
        # Handles translating positions on the image into physical distances, and then orders those movements from the drone
        # This method will assert arrival
        __park(x, y, drone)

        # Wait 5 seconds at our parking spot before returning home
        sleep(5)

    # Any failures in asserts will trickle into this finally block, and we want to land
    finally:

        # If we've saved our home position, we want to return there
        if home_lat is not None and home_long is not None:
            __move_drone(drone, home_lat, home_long, PARKING_SPOT_HOVER_HEIGHT*2)
        
        # If we had to pivot our heading to capture the parking lot, we want to reset that before landing
        if yaw_moved is not None:
            __position_camera(drone, 0, 0, 15 * yaw_moved)  # Reset our heading
        
        # Safely landy at the current lat and long, and power off propellers
        assert drone(Landing()).wait().success()
        
        # Sever the connection with the drone
        drone.disconnect()
