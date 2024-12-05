import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pygame

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,  # Single-hand interaction
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame for music playback
pygame.mixer.init()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define states
STATE_MAIN_MENU = "main_menu"
STATE_CALCULATOR = "calculator"
STATE_CLOCK = "clock"
STATE_MUSIC = "music"  # New state for Music Screen
state = STATE_MAIN_MENU

# Define screen zones and labels
zones_main_menu = [
    {"label": "Welcome", "coords": (50, 50)},
    {"label": "Calculator", "coords": (150, 50)},  # Keyboard button for Calculator
    {"label": "Clock", "coords": (250, 50)},  # Clock button for Clock Screen
    {"label": "Music", "coords": (350, 50)},  # Music button for Music Screen
]

zones_calculator = [
    {"label": "7", "coords": (100, 100)},
    {"label": "8", "coords": (200, 100)},
    {"label": "9", "coords": (300, 100)},
    {"label": "/", "coords": (400, 100)},
    {"label": "4", "coords": (100, 200)},
    {"label": "5", "coords": (200, 200)},
    {"label": "6", "coords": (300, 200)},
    {"label": "*", "coords": (400, 200)},
    {"label": "1", "coords": (100, 300)},
    {"label": "2", "coords": (200, 300)},
    {"label": "3", "coords": (300, 300)},
    {"label": "-", "coords": (400, 300)},
    {"label": "Remove", "coords": (500, 300)},
    {"label": "Clear", "coords": (500, 400)},
    {"label": "0", "coords": (200, 400)},
    {"label": "+", "coords": (300, 400)},
    {"label": "=", "coords": (400, 400)},
    {"label": "Back", "coords": (100, 400)},  # Back button to return to Main Menu
]

# Music list with file paths
songs = [
    {"label": "FE!N", "coords": (100, 100), "file": "Travis Scott - FE!N (Official Music Video) ft. Playboi Carti.mp3"},
    {"label": "Espresso", "coords": (100, 200), "file": "Sabrina Carpenter - Espresso (Official Video).mp3"},
    {"label": "Empire Of The Sun", "coords": (100, 300), "file": "Empire Of The Sun - We Are The People [HQ].mp3"},
]

zones_music = [
    {"label": "Stop", "coords": (100, 400)},
    {"label": "Rewind", "coords": (250, 400)},  # Rewind button
    {"label": "Resume", "coords": (400, 400)},  # Resume button
    {"label": "Back", "coords": (550, 400)},  # Back button to return to Main Menu
]

# Add songs to the zones_music for selection
zones_music.extend(songs)

# Variables
current_expression = ""
last_selected_zone = None
frames_on_zone = 0
selection_threshold = 25  # Frames required to select a zone
distance_threshold = 50  # Minimum distance to register a button hover (in pixels)
current_song = None  # Currently playing song

# Function to stop the song
def stop_song():
    global current_song, is_paused
    pygame.mixer.music.stop()
    current_song = None
    is_paused = False


# Function to rewind the song
def rewind_song():
    if current_song:
        pygame.mixer.music.play(start=0.0)


# Function to resume the song
def resume_song():
    global is_paused
    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False


# Function to pause the song
def pause_song():
    global is_paused
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
        is_paused = True

# Function to find the nearest zone
def find_nearest_zone(x, y, zones):
    min_distance = float("inf")
    nearest_zone = None
    for zone in zones:
        zone_x, zone_y = zone["coords"]
        distance = np.sqrt((x - zone_x) ** 2 + (y - zone_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_zone = zone["label"]
    # Only return the nearest zone if it's within the distance threshold
    if min_distance <= distance_threshold:
        return nearest_zone
    else:
        return None


# Function to play a song
def play_song(song_file):
    global current_song
    if current_song == song_file:
        return  # Do not restart if the same song is already playing
    pygame.mixer.music.load(song_file)
    pygame.mixer.music.play()
    current_song = song_file


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB as Mediapipe requires
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Determine zones based on current state
    if state == STATE_MAIN_MENU:
        zones = zones_main_menu
    elif state == STATE_CALCULATOR:
        zones = zones_calculator
    elif state == STATE_CLOCK:
        zones = [{"label": "Back", "coords": (100, 400)}]
    elif state == STATE_MUSIC:
        zones = zones_music

    # Draw zones on the screen
    for zone in zones:
        cv2.circle(frame, zone["coords"], 40, (255, 0, 0), 2)
        cv2.putText(frame, zone["label"], (zone["coords"][0] - 20, zone["coords"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the current time if in clock state
    if state == STATE_CLOCK:
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display input and output if in calculator state
    if state == STATE_CALCULATOR:
        cv2.putText(frame, f"Input: {current_expression}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Process hand landmarks and gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index fingertip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])

            # Find the nearest zone
            nearest_zone = find_nearest_zone(index_x, index_y, zones)

            if nearest_zone == last_selected_zone:
                frames_on_zone += 1
                if frames_on_zone == selection_threshold:
                    print(f"Selected: {nearest_zone}")
                    frames_on_zone = 0  # Reset the counter after selection
                    if state == STATE_MAIN_MENU:
                        if nearest_zone == "Calculator":
                            state = STATE_CALCULATOR
                        elif nearest_zone == "Clock":
                            state = STATE_CLOCK
                        elif nearest_zone == "Music":
                            state = STATE_MUSIC
                    elif state == STATE_CALCULATOR:
                        if nearest_zone == "Back":
                            state = STATE_MAIN_MENU
                            current_expression = ""
                        elif nearest_zone == "Clear":
                            current_expression = ""  # Clear the entire expression
                        elif nearest_zone == "Remove":
                            current_expression = current_expression[:-1]  # Remove the last character
                        elif nearest_zone == "=":
                            try:
                                current_expression = str(eval(current_expression))
                            except Exception:
                                current_expression = "Error"
                        else:
                            current_expression += nearest_zone
                    elif state == STATE_CLOCK:
                        if nearest_zone == "Back":
                            state = STATE_MAIN_MENU
                    elif state == STATE_MUSIC:
                        if nearest_zone == "Back":
                            state = STATE_MAIN_MENU
                            pygame.mixer.music.stop()
                            current_song = None
                        elif nearest_zone == "Stop":
                            stop_song()
                        elif nearest_zone == "Rewind":
                            rewind_song()
                        elif nearest_zone == "Resume":
                            resume_song()
                        else:
                            for song in songs:
                                if song["label"] == nearest_zone:
                                    play_song(song["file"])
                                    break
            else:
                last_selected_zone = nearest_zone
                frames_on_zone = 1  # Reset counter for a new zone

    # Display the frame
    cv2.imshow('Hand Detection with Zones', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
