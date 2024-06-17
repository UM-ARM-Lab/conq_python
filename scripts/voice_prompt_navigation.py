# export PYTHONPATH="${PYTHONPATH}:/Users/adibalaji/Desktop/agrobots/conq_python/src"

import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient

from conq.navigation.graph_nav.graph_nav_utils import GraphNav
from conq.navigation.graph_nav.scene_labeler import SceneLabeler
from conq.semantic_grasper import SemanticGrasper

import speech_recognition as sr
import time


def recognize_speech_from_mic():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the default system microphone as the audio source
    with sr.Microphone() as source:
        print("Please wait. Calibrating microphone...")
        # Listen for 5 seconds and create ambient noise energy level to reduce noise
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Microphone calibrated. Start speaking.")
        # Capture the audio from the microphone
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Web Speech API
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return


# ------------------------------------------------------------------- WITH VOICE PROMPT -----------------------------------------------------------------------------------

# Setup and authenticate the robot.
# sdk = bosdyn.client.create_standard_sdk('VoicePromptNav')
# robot = sdk.create_robot('192.168.80.3')
# bosdyn.client.util.authenticate(robot) 
# lease_client = robot.ensure_client(LeaseClient.default_service_name)
# lease_client.take()
# gn = GraphNav(robot)
# gn.localize()
# sl = SceneLabeler()
# sg = SemanticGrasper(robot)
# obj_wp_dict = sl.load_dict_from_json()
# print('Setting up microphone...')
# task = recognize_speech_from_mic()
# obj_of_interest = sl.identify_object_from_bank(task, obj_wp_dict)
# print(f'Okay! Navigating towards the {obj_of_interest}...')
# goal_waypoint = obj_wp_dict[obj_of_interest]
# gn.navigate_to(goal_waypoint, sit_down_after_reached=False)
# print(f'Reached goal! Grasping {obj_of_interest}..')
# sg.take_photos()
# object_direction = sg.find_object_in_photos(obj_of_interest)
# sg.orient_and_grasp(object_direction,obj_of_interest)
# print(f'Going back to start point...')
# gn.navigate_to('waypoint_0', sit_down_after_reached=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------- WITHOUT VOICE PROMPT -----------------------------------------------------------------------------------

sdk = bosdyn.client.create_standard_sdk('VoicePromptNav')
robot = sdk.create_robot('192.168.80.3')
bosdyn.client.util.authenticate(robot) 
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_client.take()
gn = GraphNav(robot)
gn.localize()
sl = SceneLabeler()
sg = SemanticGrasper(robot)
obj_of_interest = "drill"
print(f'Okay! Navigating towards the {obj_of_interest}...')
obj_wp_dict = sl.load_dict_from_json()
goal_waypoint = obj_wp_dict[obj_of_interest]
gn.navigate_to(goal_waypoint, sit_down_after_reached=False)
print(f'Reached goal! Searching for {obj_of_interest}..')
sg.search_object_with_gripper(obj_of_interest)
sg.orient_and_grasp('frontleft_fisheye_image', obj_of_interest)
print(f'Going back to start point...')
gn.navigate_to('waypoint_0', sit_down_after_reached=False)
sg.put_down()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    
