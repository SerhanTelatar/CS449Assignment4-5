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
    {"label": "FE!N", "file": "Travis Scott - FE!N (Official Music Video) ft. Playboi Carti.mp3"},
    {"label": "Espresso", "file": "Sabrina Carpenter - Espresso (Official Video).mp3"},
    {"label": "Empire Of The Sun", "file": "Empire Of The Sun - We Are The People [HQ].mp3"},
    {"label": "Bye Bye Bye", "file": "Bye Bye Bye.mp3"},
    {"label": "Ocean Drive", "file": "Duke Dumont - Ocean Drive.mp3"},
    {"label": "Made in Romania", "file": "Ionut Cercel - Made in Romania.mp3"},
    {"label": "Stronger", "file": "Kanye West - Stronger.mp3"},
    {"label": "League of Legends", "file": "Season 2024 Cinematic.mp3"},
    {"label": "Softcore", "file": "The Neighbourhood - Softcore.mp3"},
    {"label": "Clocks", "file": "@coldplay - Clocks.mp3"},
    {"label": "Love Again", "file": "Dua Lipa - Love Again (Lyrics).mp3"},
    {"label": "Levitating", "file": "Dua Lipa - Levitating Featuring DaBaby.mp3"},
]

zones_music = [
    {"label": "Stop", "coords": (100, 400)},
    {"label": "Rewind", "coords": (250, 400)},  # Rewind button
    {"label": "Resume", "coords": (400, 400)},  # Resume button
    {"label": "Back", "coords": (550, 400)},  # Back button to return to Main Menu
]

# Add songs to the zones_music for selection
# Not: Artık songs'u zones olarak değil, dikdörtgenleri çizerek kullanacağız.
# zones_music.extend(songs) # Bu satırı kaldırıyoruz çünkü artık songs'u kendimiz çiziyoruz.

# Volume slider properties
slider_x = 600
slider_y_start = 100
slider_y_end = 400
slider_width = 40
volume_level = 0.5  # Default volume level (50%)

# Set initial volume
pygame.mixer.music.set_volume(volume_level)

# Variables
current_expression = ""
last_selected_zone = None
frames_on_zone = 0
selection_threshold = 25  # Frames required to select a zone
distance_threshold = 50  # Minimum distance to register a button hover (in pixels)
current_song = None  # Currently playing song
is_paused = False  # Whether the music is paused

# Scroll related variables for the music list
scroll_offset = 0
song_box_width = 200
song_box_height = 50
song_start_x = 100
song_start_y = 100
song_vertical_spacing = 10
scroll_bar_x = 350
scroll_bar_width = 20
scroll_area_height = 250  # Görünür alan yüksekliği
max_scroll = max(0, (song_box_height + song_vertical_spacing)*len(songs) - scroll_area_height)

# Pinch gesture related
pinch_threshold = 30  # iki parmak arası mesafe eşiği
is_pinch_active = False
previous_pinch_y = None

def rewind_song():
    if current_song:
        pygame.mixer.music.play(start=0.0)

def resume_song():
    global is_paused
    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False

# Function to find the nearest zone (for circle zones)
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

# Function to check if a point (x,y) is inside a rectangle
def is_point_in_rect(px, py, rx, ry, rw, rh):
    return (rx <= px <= rx+rw) and (ry <= py <= ry+rh)

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

    # Draw zones on the screen (for main, calc, clock)
    # For music we'll handle songs separately
    if state != STATE_MUSIC:
        for zone in zones:
            cv2.circle(frame, zone["coords"], 40, (255, 0, 0), 2)
            cv2.putText(frame, zone["label"], (zone["coords"][0] - 20, zone["coords"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw the volume slider if in music state
    if state == STATE_MUSIC:
        # Draw the basic music control zones (Stop, Rewind, Resume, Back)
        for zone in zones_music:
            cv2.circle(frame, zone["coords"], 40, (255, 0, 0), 2)
            cv2.putText(frame, zone["label"], (zone["coords"][0] - 20, zone["coords"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw the slider background
        cv2.rectangle(frame, (slider_x, slider_y_start), (slider_x + slider_width, slider_y_end), (200, 200, 200), -1)
        # Draw the current volume level
        slider_y = int(slider_y_start + (1 - volume_level) * (slider_y_end - slider_y_start))
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_width, slider_y_end), (0, 255, 0), -1)
        cv2.putText(frame, "Volume", (slider_x - 10, slider_y_start - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the songs as rectangles with scroll offset
        # Visible area: from song_start_y to song_start_y + scroll_area_height
        current_y = song_start_y - scroll_offset
        song_rects = []
        for i, song in enumerate(songs):
            rx = song_start_x
            ry = current_y
            rw = song_box_width
            rh = song_box_height

            # Sadece görünür alana girenleri çiz
            if ry + rh > song_start_y and ry < song_start_y + scroll_area_height:
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)
                cv2.putText(frame, song["label"], (rx+10, ry+rh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                song_rects.append((song, rx, ry, rw, rh))
            current_y += (song_box_height + song_vertical_spacing)

        # Draw scrollbar
        # Scrollbar height ratio
        if max_scroll > 0:
            scroll_ratio = scroll_area_height / ((song_box_height+song_vertical_spacing)*len(songs))
        else:
            scroll_ratio = 1
        scroll_bar_height = int(scroll_area_height * scroll_ratio)
        # Scroll bar position
        scroll_pos = int((scroll_offset / max_scroll) * (scroll_area_height - scroll_bar_height)) if max_scroll > 0 else 0

        cv2.rectangle(frame, (scroll_bar_x, song_start_y), (scroll_bar_x + scroll_bar_width, song_start_y + scroll_area_height), (180,180,180), -1)
        cv2.rectangle(frame, (scroll_bar_x, song_start_y + scroll_pos), (scroll_bar_x + scroll_bar_width, song_start_y + scroll_pos + scroll_bar_height), (50,50,50), -1)

    # Process hand landmarks and gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])

            # Orta parmak ucu
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_x = int(middle_tip.x * frame.shape[1])
            middle_y = int(middle_tip.y * frame.shape[0])

            # Pinch kontrolü
            dist = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
            # Pinch aktif mi?
            currently_pinch = dist < pinch_threshold

            # Scroll işlemi (sadece müzik ekranında)
            if state == STATE_MUSIC:
                # Volume slider etkileşimi
                if slider_x <= index_x <= slider_x + slider_width and slider_y_start <= index_y <= slider_y_end:
                    volume_level = 1 - (index_y - slider_y_start) / (slider_y_end - slider_y_start)
                    volume_level = max(0, min(volume_level, 1))  # Clamp the value between 0 and 1
                    pygame.mixer.music.set_volume(volume_level)

                # Pinch ile scroll
                if currently_pinch:
                    if not is_pinch_active:
                        is_pinch_active = True
                        previous_pinch_y = index_y
                    else:
                        # Pinch zaten aktif, hareket miktarını al
                        delta_y = index_y - previous_pinch_y
                        previous_pinch_y = index_y
                        # Scroll offseti güncelle
                        scroll_offset += delta_y
                        scroll_offset = max(0, min(scroll_offset, max_scroll))
                else:
                    is_pinch_active = False
                    previous_pinch_y = None

            # Zone seçimi (main, calc, clock ve music'teki kontroller)
            if state != STATE_MUSIC:
                # Eskisi gibi en yakın zone'u bul
                nearest_zone = find_nearest_zone(index_x, index_y, zones)
            else:
                # Music ekranında temel butonlar hala circle zone
                # Bu butonlar haricinde şarkıları dikdörtgenlerden seçeceğiz.
                # Önce temel butonlar için nearest_zone:
                circle_zone = find_nearest_zone(index_x, index_y, zones_music)

                # Şarkı seçimi için dikdörtgen kontrolü
                rect_zone = None
                if state == STATE_MUSIC and 'song_rects' in locals():
                    for song, rx, ry, rw, rh in song_rects:
                        if is_point_in_rect(index_x, index_y, rx, ry, rw, rh):
                            rect_zone = song["label"]
                            break

                # Öncelik şarkı dikdörtgenlerinde
                if rect_zone is not None:
                    nearest_zone = rect_zone
                else:
                    nearest_zone = circle_zone

            if nearest_zone == last_selected_zone:
                frames_on_zone += 1
                if frames_on_zone == selection_threshold and nearest_zone is not None:
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
                            pygame.mixer.music.stop()
                            current_song = None
                        elif nearest_zone == "Rewind":
                            rewind_song()
                        elif nearest_zone == "Resume":
                            resume_song()
                        else:
                            # Eğer şarkı seçimiyse:
                            for s in songs:
                                if s["label"] == nearest_zone:
                                    play_song(s["file"])
                                    break
            else:
                last_selected_zone = nearest_zone
                frames_on_zone = 1  # Reset counter for a new zone

    # Calculator expression display
    if state == STATE_CALCULATOR:
        cv2.putText(frame, current_expression, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # Clock display
    if state == STATE_CLOCK:
        now = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, now, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    # Draw a rectangle to represent the visible area of songs in music mode (for clarity)
    if state == STATE_MUSIC:
        cv2.rectangle(frame, (song_start_x-10, song_start_y), (song_start_x+song_box_width+10, song_start_y+scroll_area_height), (255,255,255), 1)

    # Display the frame
    cv2.imshow('Hand Detection with Zones', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()