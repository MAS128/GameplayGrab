Screen + Input Capture for Machine Learning Datasets

This Python script captures screen frames alongside user input (keyboard and mouse events) to create a dataset for machine learning or analytics. Each trial produces:
	1.	Images (frames) stored in a folder (e.g., Trial_0).
	2.	A JSON log (TrialData_0.json) with timestamps of frames and events (keys pressed, mouse position, etc.).

Features
	•	Frame Capture: Uses mss to grab the primary screen at ~30 FPS.
	•	Keyboard & Mouse Events: Logged via pynput for standard OS-level input (positions, clicks, keys).
	•	Optional Raw Input (macOS, will add windows later):
	•	When the script starts, it prompts Use raw mouse data via Quartz? (y/n) (only macos).
	•	Trials & Logging:
	•	Each trial is stored in a Trial_<N> folder.
	•	A JSON file TrialData_<N>.json keeps the event+frame metadata (timestamps, keys, mouse deltas, etc.).
	•	Visualization:
	•	You can replay frames in a simple OpenCV window and view the associated events in the console.

Requirements
	•	Python 3.7+
	•	mss
	•	pynput
	•	rich
	•	opencv-python

Usage

	1.	Install Dependencies:

pip install mss pynput rich opencv-python

	2.	Run the Script:

python gg.py

	3.	Choose Dataset Name: You’ll be prompted:

Enter dataset name:

This creates/uses a folder, scans existing trials, and sets up the next trial number.

	4.	macOS Raw Input Prompt (if on Darwin):
 
 Use raw mouse data via Quartz? (y/n):
	•	y: sets up the Quartz event tap for raw deltas.
	•	n: uses only standard OS input from pynput.
 
	5.	Commands (after dataset is ready):
 
	•	S = Start a new trial (capture frames + inputs).
	•	Q = Stop the current trial (save JSON log, end screen capture).
	•	QQ = Exit the program entirely.
	•	N = Reset (choose a new dataset folder).
	•	V = Visualize a trial. You’ll be asked which trial number (or blank for latest).
	•	In the OpenCV “Visualization” window, arrow keys navigate frames.
	•	In the console, type q to exit visualization mode.
 
	6.	Check Output:
 
	•	Images in Trial_<N> folder (frame_0.webp, frame_1.webp, etc.).
	•	TrialData_<N>.json with all events and frame references.

Notes & Limitations
	•	macOS Fullscreen Games: Even with raw input taps, many fullscreen games don’t propagate real deltas. You may see 0 if the game exclusively locks the mouse, or frozen x and y.
	•	Permissions: On macOS, ensure you grant Accessibility (and possibly Screen Recording / Input Monitoring) to your Python or Terminal app.
	•	Data Overload: If you hold a key/mouse for a long time, the script WILL record frames at rate of 30 FPS.
	•	Frame Rate: Defaults to 30 FPS, adjustable in the code.
