import os
import cv2
import mss
import time
import json
import numpy as np
import platform
from threading import Thread, Lock, Event
from pynput import keyboard, mouse

from rich.console import Console
from rich.table import Table

# -----------------------------------------------------------
# OS DETECTION
# -----------------------------------------------------------
OS_NAME = platform.system()
print("OS Detected:", OS_NAME)

USE_MAC_RAW_INPUT = False
if OS_NAME == "Darwin":
    print("[INFO] macOS detected.")
    print("[???]: Use raw mouse capture? [Y/N]")
else:
    print("[INFO] Raw mouse capture only works on macOS (Quartz). Using normal approach on", OS_NAME)

# We'll store whether the user wants raw data here
enable_mac_raw_data = False

# -----------------------------------------------------------
# MAC-SPECIFIC RAW INPUT (Quartz)
# -----------------------------------------------------------
raw_mouse_deltas = []  # We'll store raw events {timestamp, dx, dy} here

def mouse_event_callback(proxy, event_type, event, refcon):
    """
    Low-level Quartz callback for mouse movement (including raw deltas).
    """
    import Quartz
    dx = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaX)
    dy = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaY)
    timestamp = time.time()
    
    # Append to a global list
    raw_mouse_deltas.append({
        "timestamp": timestamp,
        "dx": dx,
        "dy": dy
    })
    return event

def start_mac_raw_input_tap():
    """
    Creates an event tap at the Quartz level to intercept mouse movements.
    Requires 'Security & Privacy -> Accessibility' permission on macOS.
    """
    import Quartz
    mask = (   Quartz.kCGEventMouseMoved
            | Quartz.kCGEventLeftMouseDragged
            | Quartz.kCGEventRightMouseDragged
            | Quartz.kCGEventOtherMouseDragged )

    # Use kCGHIDEventTap for lower-level hooking
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGHIDEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        mask,
        mouse_event_callback,
        None
    )

    if not tap:
        print("[ERROR] Failed to create event tap. Check permissions or code signing.")
        return

    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(
        Quartz.CFRunLoopGetCurrent(),
        run_loop_source,
        Quartz.kCFRunLoopCommonModes
    )

    Quartz.CGEventTapEnable(tap, True)

    def run_loop_thread():
        Quartz.CFRunLoopRun()

    t = Thread(target=run_loop_thread, daemon=True)
    t.start()
    print("[INFO] macOS raw input tap started.")

# -----------------------------------------------------------
# GLOBAL CONFIGURATIONS
# -----------------------------------------------------------
FRAME_WIDTH, FRAME_HEIGHT = 480, 480
FPS = 30
MOUSE_MOVE_TIMEOUT = 0.2

SAVE_FORMAT = ".webp"
SAVE_QUALITY_PARAM = cv2.IMWRITE_WEBP_QUALITY
SAVE_QUALITY_VALUE = 70

# Key codes for arrow navigation in the OpenCV window
LEFT_KEYS = [2424832, 65361, 63234]   # Left
RIGHT_KEYS = [2555904, 65363, 63235]  # Right
UP_KEYS = [2490368, 65362, 63232]     # Up
DOWN_KEYS = [2621440, 65364, 63233]   # Down

# -----------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------
running = True
recording = False
dataset_name = None
next_trial_number = 0
trial_data_log = []
trial_frame_counter = 0
trial_folder_path = None

pressed_keys = set()
pressed_mouse_buttons = set()

# Cursor tracking (pynput)
last_mouse_move_time = 0.0
last_mouse_x = 0.0
last_mouse_y = 0.0

# For the mac raw input poll
mac_raw_poll_thread = None

# Synchronization
state_lock = Lock()
stop_capture_event = Event()

# -----------------------------------------------------------
# EVENTS STORAGE
# -----------------------------------------------------------
pending_events = []
global_event_count = 0

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def ensure_folder_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def scan_existing_trials(dataset_path: str) -> int:
    max_trial = -1
    for entry in os.listdir(dataset_path):
        if entry.startswith("Trial_"):
            try:
                trial_num = int(entry.replace("Trial_", ""))
                if trial_num > max_trial:
                    max_trial = trial_num
            except ValueError:
                pass
    return max_trial

def save_trial_data_log(trial_number: int):
    global trial_data_log, dataset_name
    if dataset_name is None:
        return

    json_filename = os.path.join(dataset_name, f"TrialData_{trial_number}.json")
    with open(json_filename, "w") as f:
        json.dump(trial_data_log, f, indent=4)
    print(f"[INFO] Trial #{trial_number} data saved to {json_filename}")

def add_event(keyboard_keys=None, mouse_buttons=None, 
              mouse_move=False, x=None, y=None, dx=None, dy=None,
              raw_input=False):
    """
    Adds a new event to 'pending_events'.
    'raw_input' indicates if it's from mac raw deltas (Quartz), or normal OS data.
    """
    global global_event_count
    evt = {
        "global_number": global_event_count,
        "timestamp": time.time(),
        "keyboard_keys": keyboard_keys or [],
        "mouse_buttons": mouse_buttons or [],
        "mouse_is_moving": mouse_move,
        "raw_input": raw_input
    }
    if mouse_move:
        if x is not None and y is not None:
            evt["mouse_xy"] = [x, y]
        if dx is not None and dy is not None:
            evt["mouse_delta"] = [dx, dy]

    global_event_count += 1
    pending_events.append(evt)

# -----------------------------------------------------------
# MAC RAW INPUT POLLING
# -----------------------------------------------------------
def poll_mac_raw_deltas():
    """
    Continuously poll the 'raw_mouse_deltas' queue and convert them into events.
    This thread runs only while 'recording' is True.
    """
    while recording:
        time.sleep(0.01)
        with state_lock:
            while raw_mouse_deltas:
                delta = raw_mouse_deltas.pop(0)
                dx = delta["dx"]
                dy = delta["dy"]
                add_event(
                    mouse_move=True, 
                    dx=dx, 
                    dy=dy,
                    raw_input=True
                )

# -----------------------------------------------------------
# PYNPUT EVENT HANDLERS (absolute OS cursor)
# -----------------------------------------------------------
last_os_mouse_x = 0.0
last_os_mouse_y = 0.0

def on_key_press(key):
    if not recording:
        return
    try:
        key_name = key.char if key.char else str(key)
    except:
        key_name = str(key)
    with state_lock:
        if key_name not in pressed_keys:
            pressed_keys.add(key_name)
            add_event(keyboard_keys=[key_name], mouse_move=False)

def on_key_release(key):
    if not recording:
        return
    try:
        key_name = key.char if key.char else str(key)
    except:
        key_name = str(key)
    with state_lock:
        if key_name in pressed_keys:
            pressed_keys.remove(key_name)
            add_event(keyboard_keys=[key_name], mouse_move=False)

def on_mouse_click(x, y, button, pressed):
    if not recording:
        return
    btn_str = str(button)
    with state_lock:
        if pressed:
            if btn_str not in pressed_mouse_buttons:
                pressed_mouse_buttons.add(btn_str)
            add_event(mouse_buttons=[btn_str], mouse_move=False)
        else:
            if btn_str in pressed_mouse_buttons:
                pressed_mouse_buttons.remove(btn_str)
            add_event(mouse_buttons=[btn_str], mouse_move=False)

def on_mouse_move(x, y):
    if not recording:
        return
    global last_mouse_move_time, last_mouse_x, last_mouse_y
    global last_os_mouse_x, last_os_mouse_y

    now = time.time()
    dx = x - last_os_mouse_x
    dy = y - last_os_mouse_y

    with state_lock:
        last_os_mouse_x, last_os_mouse_y = x, y
        last_mouse_x, last_mouse_y = float(x), float(y)
        last_mouse_move_time = now
        add_event(
            mouse_move=True,
            x=last_mouse_x,
            y=last_mouse_y,
            dx=dx,
            dy=dy,
            raw_input=False
        )

# -----------------------------------------------------------
# CAPTURE FUNCTION (SCREEN + EVENTS)
# -----------------------------------------------------------
def capture_screen():
    global trial_frame_counter, trial_data_log
    global pressed_keys, pressed_mouse_buttons
    global last_mouse_move_time, last_mouse_x, last_mouse_y
    global dataset_name, trial_folder_path, pending_events
    global recording

    last_frame_time = time.time()
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary screen

        while not stop_capture_event.is_set():
            frame_start = time.time()
            with state_lock:
                mouse_is_moving = (frame_start - last_mouse_move_time) < MOUSE_MOVE_TIMEOUT
                should_capture = bool(pressed_keys or pressed_mouse_buttons or mouse_is_moving)

            if should_capture:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                frame_time = time.time()
                frame_events = []
                local_idx = 0

                with state_lock:
                    i = 0
                    while i < len(pending_events):
                        evt = pending_events[i]
                        if last_frame_time < evt["timestamp"] <= frame_time:
                            new_evt = dict(evt)
                            new_evt["number"] = local_idx
                            local_idx += 1
                            frame_events.append(new_evt)
                            pending_events.pop(i)
                        else:
                            i += 1

                    frame_filename = f"frame_{trial_frame_counter}{SAVE_FORMAT}"
                    full_frame_path = os.path.join(trial_folder_path, frame_filename)
                    cv2.imwrite(full_frame_path, img, [SAVE_QUALITY_PARAM, SAVE_QUALITY_VALUE])

                    frame_entry = {
                        "filename": frame_filename,
                        "timestamp": frame_time,
                        "events": frame_events,
                        "held_keys": list(pressed_keys),
                        "held_buttons": list(pressed_mouse_buttons),
                        "mouse_is_moving": mouse_is_moving,
                        "mouse_xy": [last_mouse_x, last_mouse_y]
                    }

                    trial_data_log.append(frame_entry)
                    trial_frame_counter += 1

                last_frame_time = frame_time

            elapsed = time.time() - frame_start
            sleep_time = (1.0 / FPS) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cv2.destroyAllWindows()

# -----------------------------------------------------------
# VISUALIZATION MODE
# -----------------------------------------------------------
vis_running = False
stop_visualization_event = Event()

def visualization_console_listener():
    global vis_running
    while vis_running:
        cmd = input()
        if cmd.strip().lower() == 'q':
            vis_running = False
            stop_visualization_event.set()
            print("[INFO] Exiting visualization mode.")

def visualize_dataset():
    global vis_running, dataset_name

    trial_input = input("Enter trial number to visualize (leave blank for last trial): ").strip()
    if trial_input == "":
        chosen_trial = scan_existing_trials(dataset_name)
        if chosen_trial < 0:
            print("[WARN] No trials found in this dataset. Nothing to visualize.")
            return
    else:
        try:
            chosen_trial = int(trial_input)
        except ValueError:
            print("[WARN] Invalid trial number input. Aborting visualization.")
            return

    json_path = os.path.join(dataset_name, f"TrialData_{chosen_trial}.json")
    if not os.path.exists(json_path):
        print(f"[WARN] JSON for trial {chosen_trial} not found at: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = [e for e in data if "filename" in e]
    if not frames:
        print(f"[WARN] No frame entries found in trial {chosen_trial}. Nothing to visualize.")
        return

    trial_folder = os.path.join(dataset_name, f"Trial_{chosen_trial}")
    if not os.path.isdir(trial_folder):
        print(f"[WARN] Folder for trial {chosen_trial} not found: {trial_folder}")
        return

    cv2.namedWindow("Visualization", cv2.WINDOW_NORMAL)

    idx = 0
    total_frames = len(frames)

    vis_thread = Thread(target=visualization_console_listener, daemon=True)
    vis_running = True
    stop_visualization_event.clear()
    vis_thread.start()

    console = Console()
    last_rendered_idx = None

    while vis_running and not stop_visualization_event.is_set():
        key = cv2.waitKeyEx(50)
        if key in LEFT_KEYS:
            idx -= 1
        elif key in RIGHT_KEYS:
            idx += 1
        elif key in UP_KEYS:
            idx += 30
        elif key in DOWN_KEYS:
            idx -= 30

        if idx < 0:
            idx = 0
        if idx >= total_frames:
            idx = total_frames - 1

        if idx != last_rendered_idx:
            last_rendered_idx = idx
            fdata = frames[idx]
            frame_file = os.path.join(trial_folder, fdata["filename"])
            if not os.path.exists(frame_file):
                print(f"[WARN] Frame not found: {frame_file}")
                break

            img = cv2.imread(frame_file)
            if img is None:
                print(f"[WARN] Could not read frame: {frame_file}")
                break

            # Make space at bottom for text
            label_height = 30
            extended = np.zeros((img.shape[0] + label_height, img.shape[1], 3), dtype=np.uint8)
            extended[:img.shape[0], :img.shape[1]] = img

            info_text = f"{fdata['filename']} | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fdata['timestamp']))}"
            cv2.putText(
                extended, info_text,
                (10, img.shape[0] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
            )
            cv2.imshow("Visualization", extended)

            console.clear()
            frame_ts_human = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fdata["timestamp"]))
            console.print(
                f"[bold green]{frame_ts_human} | Frame {idx+1}/{total_frames} | {fdata['filename']}[/bold green]"
            )

            # Build events table
            if fdata["events"]:
                table = Table(title="Events in this frame", header_style="bold cyan")
                table.add_column("local_number", style="dim")
                table.add_column("timestamp", justify="right")
                table.add_column("global_number", justify="right")
                table.add_column("keyboard_keys", justify="left")
                table.add_column("mouse_buttons", justify="left")
                table.add_column("mouse_is_moving", justify="center")
                table.add_column("mouse_xy", justify="center")
                table.add_column("mouse_delta", justify="center")
                table.add_column("raw_input", justify="center")

                for evt in fdata["events"]:
                    loc_num = str(evt.get("number", "--"))
                    gnum = str(evt.get("global_number", "--"))
                    evt_ts = f"{evt.get('timestamp', '--'):.4f}" if "timestamp" in evt else "--"
                    kb = str(evt.get("keyboard_keys", [])) or "--"
                    mb = str(evt.get("mouse_buttons", [])) or "--"
                    mm = str(evt.get("mouse_is_moving", "--"))
                    mxy = str(evt.get("mouse_xy", "--"))
                    mdelta = str(evt.get("mouse_delta", "--"))
                    raw = str(evt.get("raw_input", False))

                    table.add_row(loc_num, evt_ts, gnum, kb, mb, mm, mxy, mdelta, raw)

                console.print(table)
            else:
                console.print("[dim]-- no events --[/dim]")

            # Additional "Frame State"
            state_table = Table(title="Frame State at Capture", header_style="bold magenta")
            state_table.add_column("held_keys", justify="left")
            state_table.add_column("held_buttons", justify="left")
            state_table.add_column("mouse_is_moving", justify="center")
            state_table.add_column("mouse_xy", justify="center")

            hk_str = str(fdata.get("held_keys", []))
            hb_str = str(fdata.get("held_buttons", []))
            mm_str = str(fdata.get("mouse_is_moving", "--"))
            xy_str = str(fdata.get("mouse_xy", "--"))

            state_table.add_row(hk_str, hb_str, mm_str, xy_str)
            console.print(state_table)

    cv2.destroyWindow("Visualization")
    console.clear()
    print("[INFO] Visualization mode ended.")

# -----------------------------------------------------------
# CONSOLE COMMAND HANDLERS
# -----------------------------------------------------------
def initialize_dataset():
    global dataset_name, next_trial_number
    while True:
        name = input("Enter dataset name: ").strip()
        if not name:
            print("Dataset name cannot be empty.")
            continue

        path = os.path.abspath(name)
        ensure_folder_exists(path)

        last_trial = scan_existing_trials(path)
        if last_trial >= 0:
            next_trial_number = last_trial + 1
            print(f"[INFO] Found existing dataset with {last_trial+1} trials. Next trial is #{next_trial_number}.")
        else:
            next_trial_number = 0
            print("[INFO] No existing trials found. Starting at trial #0.")

        dataset_name = path
        print(f"[INFO] Dataset folder is ready: {dataset_name}")
        break

def start_trial():
    global recording, trial_data_log, trial_frame_counter
    global dataset_name, next_trial_number, trial_folder_path
    global stop_capture_event, pending_events, global_event_count
    global mac_raw_poll_thread

    if not dataset_name:
        print("[WARN] No dataset initialized. Use 'N' to set it up.")
        return
    if recording:
        print("[WARN] Already recording. Stop first with 'Q'.")
        return

    trial_folder_path = os.path.join(dataset_name, f"Trial_{next_trial_number}")
    ensure_folder_exists(trial_folder_path)

    trial_data_log.clear()
    trial_frame_counter = 0
    with state_lock:
        pending_events.clear()
        global_event_count = 0

    trial_data_log.append({
        "type": "trial_start",
        "timestamp": time.time(),
        "trial_number": next_trial_number
    })

    stop_capture_event.clear()
    recording = True

    # Start screen capture thread
    t = Thread(target=capture_screen, daemon=True)
    t.start()

    # If on mac and user enabled raw data, also start the event tap + a poll thread
    if (OS_NAME == "Darwin") and (enable_mac_raw_data):
        start_mac_raw_input_tap()
        mac_raw_poll_thread = Thread(target=poll_mac_raw_deltas, daemon=True)
        mac_raw_poll_thread.start()

    print(f"[INFO] Started trial #{next_trial_number} in: {trial_folder_path}")

def stop_trial():
    global recording, trial_data_log, next_trial_number
    global stop_capture_event
    if not recording:
        print("[WARN] No trial is recording. Use 'S' to start a trial.")
        return

    stop_capture_event.set()
    recording = False

    trial_data_log.append({
        "type": "trial_end",
        "timestamp": time.time(),
        "trial_number": next_trial_number
    })

    time.sleep(0.5)
    save_trial_data_log(next_trial_number)
    print(f"[INFO] Trial #{next_trial_number} ended.")
    next_trial_number += 1

def exit_program():
    global running
    if recording:
        print("[WARN] Stop current trial before exiting.")
        return
    print("[INFO] Exiting program.")
    running = False

def reset_program():
    global dataset_name, next_trial_number, recording
    if recording:
        print("[WARN] Stop current trial before resetting.")
        return

    dataset_name = None
    next_trial_number = 0
    initialize_dataset()

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    global running, recording, enable_mac_raw_data

    # 1) If on macOS, ask user if they want raw data
    if OS_NAME == "Darwin":
        raw_choice = input("Use raw mouse data via Quartz? (y/n): ").strip().lower()
        if raw_choice == "y":
            enable_mac_raw_data = True
            print("[INFO] Raw data capture from Quartz will be used.")
        else:
            print("[INFO] Using normal Pynput-based approach only.")
    else:
        print("[INFO] Not macOS, skipping raw data question.")

    # 2) Start normal global keyboard/mouse listeners
    kb_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    ms_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click)
    kb_listener.start()
    ms_listener.start()

    # 3) Initialize dataset
    initialize_dataset()

    while running:
        cmd = input("Enter command [S=Start, Q=Stop, QQ=Exit, N=Reset, V=Visualize]: ").strip().lower()
        if cmd == "s":
            start_trial()
        elif cmd == "q":
            stop_trial()
            cv2.destroyAllWindows()
        elif cmd == "qq":
            exit_program()
        elif cmd == "n":
            reset_program()
        elif cmd == "v":
            visualize_dataset()
        else:
            print(f"[WARN] Unknown command: {cmd}")

    kb_listener.stop()
    ms_listener.stop()
    kb_listener.join()
    ms_listener.join()

    print("[INFO] Program has exited.")

if __name__ == "__main__":
    main()