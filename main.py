import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pygame

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

pygame.mixer.init()

cursor_img = cv2.imread('download.png', cv2.IMREAD_UNCHANGED)
cursor_img = cv2.resize(cursor_img, (30, 30), interpolation=cv2.INTER_AREA)
hover_img = cv2.imread('hover.png', cv2.IMREAD_UNCHANGED)
hover_img = cv2.resize(hover_img, (30, 30), interpolation=cv2.INTER_AREA)

def overlay_image_alpha(img, img_overlay, x, y):
    if img_overlay is None:
        return
    y1, y2 = max(0,y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0,x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0,-y), min(img_overlay.shape[0], img.shape[0]-y)
    x1o, x2o = max(0,-x), min(img_overlay.shape[1], img.shape[1]-x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    overlay = img_overlay[y1o:y2o, x1o:x2o]
    if overlay.shape[2] == 4:
        overlay_img = overlay[:,:,:3]
        mask = overlay[:,:,3:] / 255.0
        img[y1:y2, x1:x2, :3] = (1.0 - mask)*img[y1:y2, x1:x2,:3] + mask * overlay_img
    else:
        img[y1:y2, x1:x2] = overlay

cap = cv2.VideoCapture(0)

STATE_MAIN_MENU = "main_menu"
STATE_CALCULATOR = "calculator"
STATE_CLOCK = "clock"
STATE_MUSIC = "music"
state = STATE_MAIN_MENU

zones_main_menu = [
    {"label": "Welcome", "coords": (50, 50)},
    {"label": "Calculator", "coords": (150, 50)},
    {"label": "Clock", "coords": (250, 50)},
    {"label": "Music", "coords": (350, 50)},
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
    {"label": "Back", "coords": (100, 400)},
]

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
    {"label": "Rewind", "coords": (250, 400)},
    {"label": "Resume", "coords": (400, 400)},
    {"label": "Back", "coords": (550, 400)},
]

slider_x = 600
slider_y_start = 100
slider_y_end = 400
slider_width = 40
volume_level = 0.5
pygame.mixer.music.set_volume(volume_level)

current_expression = ""
last_selected_zone = None
frames_on_zone = 0
selection_threshold = 25
distance_threshold = 50
current_song = None
is_paused = False

scroll_offset = 0
song_box_width = 200
song_box_height = 50
song_start_x = 100
song_start_y = 100
song_vertical_spacing = 10
scroll_bar_x = 350
scroll_bar_width = 20
scroll_area_height = 250
max_scroll = max(0, (song_box_height + song_vertical_spacing)*len(songs) - scroll_area_height)

pinch_threshold = 30
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

def find_nearest_zone(x, y, zones):
    min_distance = float("inf")
    nearest_zone = None
    for zone in zones:
        zone_x, zone_y = zone["coords"]
        distance = np.sqrt((x - zone_x) ** 2 + (y - zone_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_zone = zone["label"]
    if min_distance <= distance_threshold:
        return nearest_zone
    else:
        return None

def is_point_in_rect(px, py, rx, ry, rw, rh):
    return (rx <= px <= rx+rw) and (ry <= py <= ry+rh)

def play_song(song_file):
    global current_song
    if current_song == song_file:
        return
    pygame.mixer.music.load(song_file)
    pygame.mixer.music.play()
    current_song = song_file

def draw_button(frame, label, coords, hovered=False, selected=False):
    color = (255,0,0)
    thickness = 2
    if hovered:
        color = (0,255,0)
        thickness = 3
    if selected:
        color = (0,255,255)
        thickness = 4

    cv2.circle(frame, coords, 40, color, thickness)
    cv2.putText(frame, label, (coords[0] - 20, coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_song_item(frame, song, rx, ry, rw, rh, hovered=False):
    color = (0,0,255)
    thickness = 2
    if hovered:
        color = (0,255,0)
        thickness = 3
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, thickness)
    cv2.putText(frame, song["label"], (rx+10, ry+rh//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if state == STATE_MAIN_MENU:
        zones = zones_main_menu
    elif state == STATE_CALCULATOR:
        zones = zones_calculator
    elif state == STATE_CLOCK:
        zones = [{"label": "Back", "coords": (100, 400)}]
    elif state == STATE_MUSIC:
        zones = zones_music

    if state != STATE_MUSIC:
        for zone in zones:
            draw_button(frame, zone["label"], zone["coords"], hovered=False, selected=False)
    else:
        for zone in zones_music:
            draw_button(frame, zone["label"], zone["coords"], hovered=False, selected=False)

        cv2.rectangle(frame, (slider_x, slider_y_start), (slider_x + slider_width, slider_y_end), (200, 200, 200), -1)
        slider_y = int(slider_y_start + (1 - volume_level) * (slider_y_end - slider_y_start))
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_width, slider_y_end), (0, 255, 0), -1)
        cv2.putText(frame, "Volume", (slider_x - 10, slider_y_start - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_y = song_start_y - scroll_offset
        song_rects = []
        for i, song in enumerate(songs):
            rx = song_start_x
            ry = current_y
            rw = song_box_width
            rh = song_box_height
            if ry + rh > song_start_y and ry < song_start_y + scroll_area_height:
                draw_song_item(frame, song, rx, ry, rw, rh, hovered=False)
                song_rects.append((song, rx, ry, rw, rh))
            current_y += (song_box_height + song_vertical_spacing)

        if max_scroll > 0:
            scroll_ratio = scroll_area_height / ((song_box_height+song_vertical_spacing)*len(songs))
        else:
            scroll_ratio = 1
        scroll_bar_height = int(scroll_area_height * scroll_ratio)
        scroll_pos = int((scroll_offset / max_scroll) * (scroll_area_height - scroll_bar_height)) if max_scroll > 0 else 0
        cv2.rectangle(frame, (scroll_bar_x, song_start_y), (scroll_bar_x + scroll_bar_width, song_start_y + scroll_area_height), (180,180,180), -1)
        cv2.rectangle(frame, (scroll_bar_x, song_start_y + scroll_pos), (scroll_bar_x + scroll_bar_width, song_start_y + scroll_pos + scroll_bar_height), (50,50,50), -1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])

            overlay_image_alpha(frame, cursor_img, index_x, index_y)

            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_x = int(middle_tip.x * frame.shape[1])
            middle_y = int(middle_tip.y * frame.shape[0])

            dist = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
            currently_pinch = dist < pinch_threshold

            if state == STATE_MUSIC:
                if slider_x <= index_x <= slider_x + slider_width and slider_y_start <= index_y <= slider_y_end:
                    volume_level = 1 - (index_y - slider_y_start) / (slider_y_end - slider_y_start)
                    volume_level = max(0, min(volume_level, 1))
                    pygame.mixer.music.set_volume(volume_level)

                if currently_pinch:
                    if not is_pinch_active:
                        is_pinch_active = True
                        previous_pinch_y = index_y
                    else:
                        delta_y = index_y - previous_pinch_y
                        previous_pinch_y = index_y
                        scroll_offset += delta_y
                        scroll_offset = max(0, min(scroll_offset, max_scroll))
                else:
                    is_pinch_active = False
                    previous_pinch_y = None

            if state != STATE_MUSIC:
                nearest_zone = find_nearest_zone(index_x, index_y, zones)
            else:
                circle_zone = find_nearest_zone(index_x, index_y, zones_music)
                rect_zone = None
                hovered_song = None
                if state == STATE_MUSIC and 'song_rects' in locals():
                    for s, rx, ry, rw, rh in song_rects:
                        if is_point_in_rect(index_x, index_y, rx, ry, rw, rh):
                            rect_zone = s["label"]
                            hovered_song = (s, rx, ry, rw, rh)
                            break
                if rect_zone is not None:
                    nearest_zone = rect_zone
                else:
                    nearest_zone = circle_zone

            if nearest_zone is not None:
                overlay_image_alpha(frame, hover_img, index_x, index_y)

            # Redraw hovered elements with hover effect
            if state != STATE_MUSIC:
                for zone in zones:
                    hovered = (zone["label"] == nearest_zone)
                    selected = (zone["label"] == last_selected_zone and frames_on_zone >= selection_threshold)
                    draw_button(frame, zone["label"], zone["coords"], hovered=hovered, selected=selected)
            else:
                for zone in zones_music:
                    hovered = (zone["label"] == nearest_zone)
                    selected = (zone["label"] == last_selected_zone and frames_on_zone >= selection_threshold)
                    draw_button(frame, zone["label"], zone["coords"], hovered=hovered, selected=selected)

                for s, rx, ry, rw, rh in song_rects:
                    hovered = (s["label"] == nearest_zone)
                    draw_song_item(frame, s, rx, ry, rw, rh, hovered=hovered)

                if scroll_bar_x <= index_x <= scroll_bar_x + scroll_bar_width and song_start_y <= index_y <= song_start_y + scroll_area_height:
                    cv2.rectangle(frame, (scroll_bar_x, song_start_y), (scroll_bar_x + scroll_bar_width, song_start_y + scroll_area_height), (0,255,0), 2)

            # ADDED: Draw circular timer when hovering.
            # If we are hovering over the same zone, we increase frames_on_zone
            # and draw a partial arc representing timer progress.
            if nearest_zone == last_selected_zone:
                frames_on_zone += 1
                # Draw the timer arc only if we haven't selected yet
                if nearest_zone is not None and frames_on_zone < selection_threshold:
                    # Calculate arc angle
                    angle = int((frames_on_zone / selection_threshold) * 360)
                    # Find the coords of the hovered zone to draw the arc
                    # Try to find the zone in current zones
                    hovered_coords = None
                    if state != STATE_MUSIC:
                        for zone in zones:
                            if zone["label"] == nearest_zone:
                                hovered_coords = zone["coords"]
                                break
                    else:
                        # Check music controls
                        found = False
                        for zone in zones_music:
                            if zone["label"] == nearest_zone:
                                hovered_coords = zone["coords"]
                                found = True
                                break
                        # If not found in music controls, it might be a song
                        if not found and hovered_song is not None:
                            # hover over a song; draw arc around center of the song box
                            s, rx, ry, rw, rh = hovered_song
                            hovered_coords = (rx + rw//2, ry + rh//2)
                    
                    if hovered_coords is not None:
                        # Draw the arc as a partial ellipse (circle)
                        # We'll draw a white arc growing clockwise
                        cv2.ellipse(frame, hovered_coords, (50,50), 0, 0, angle, (255,255,255), 3)
                    
                if frames_on_zone == selection_threshold and nearest_zone is not None:
                    print(f"Selected: {nearest_zone}")
                    frames_on_zone = 0
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
                            current_expression = ""
                        elif nearest_zone == "Remove":
                            current_expression = current_expression[:-1]
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
                            for s in songs:
                                if s["label"] == nearest_zone:
                                    play_song(s["file"])
                                    break
            else:
                last_selected_zone = nearest_zone
                frames_on_zone = 1

    if state == STATE_CALCULATOR:
        cv2.putText(frame, current_expression, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if state == STATE_CLOCK:
        now = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, now, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

    if state == STATE_MUSIC:
        cv2.rectangle(frame, (song_start_x-10, song_start_y), (song_start_x+song_box_width+10, song_start_y+scroll_area_height), (255,255,255), 1)

    cv2.imshow('Hand Detection with Zones', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
