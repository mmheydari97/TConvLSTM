import numpy as np
from datetime import datetime
import os

# --- Configuration ---

# Sensor mapping to 2D grid coordinates (row, col)
# Based on aruba.jpg and preprocess.py
# Grid dimensions will be 10 rows, 8 columns (indices 0-9 for rows, 0-7 for columns)
SENSOR_COORDINATES = {
    "M001": (8, 5), "M002": (8, 3), "M003": (7, 3), "M004": (6, 6),
    "M005": (6, 4), "M006": (6, 3), "M007": (7, 5), "M008": (6, 2),
    "M009": (6, 1), "M010": (7, 1), "M011": (8, 2), "M012": (7, 0),
    "M013": (6, 0), "M014": (4, 0), "M015": (3, 0), "M016": (1, 1),
    "M017": (2, 1), "M018": (3, 1), "M019": (2, 0), "M020": (5, 1), # M020 estimated from map
    "M021": (5, 2), "M022": (5, 4), "M023": (4, 5), "M024": (4, 6),
    "M025": (2, 7), "M026": (1, 7), "M027": (1, 6), # M027 estimated, M026 from preprocess
    "M028": (3, 5), "M029": (2, 4), "M030": (1, 4), "M031": (3, 2),
}
GRID_HEIGHT = 10
GRID_WIDTH = 8

# --- Helper Functions ---

def parse_datetime_flexible(date_str, time_str):
    """
    Parses datetime string, trying with and without microseconds.
    """
    try:
        # Try with microseconds
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # Try without microseconds
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

def load_and_filter_log_data(log_filepath, chunk_size=5000): # chunk_size is illustrative for the concept
    """
    Reads the log file line by line (efficient for large files), 
    filters for motion sensors (M), and parses relevant information.
    Returns a sorted list of events: (datetime_obj, sensor_id, state_str).
    The chunk_size parameter is not directly used for reading in fixed N-line chunks here,
    as line-by-line processing is generally more memory efficient for parsing.
    It's kept to acknowledge the user's phrasing but the implementation reads one line at a time.
    """
    raw_events = []
    if not os.path.exists(log_filepath):
        print(f"Error: Log file not found at {log_filepath}")
        return raw_events

    print(f"Processing log file: {log_filepath} line by line...")
    line_count = 0
    processed_motion_events = 0
    with open(log_filepath, 'r') as f:
        for line in f: # Read file line by line
            line_count += 1
            parts = line.strip().split()
            if len(parts) < 4:
                # print(f"Skipping malformed line {line_count}: {line.strip()}")
                continue # Skip malformed lines

            date_str, time_str, sensor_id, state_str = parts[0], parts[1], parts[2], parts[3]

            if sensor_id.startswith('M'): # Filter for motion sensors
                try:
                    dt_obj = parse_datetime_flexible(date_str, time_str)
                    raw_events.append((dt_obj, sensor_id, state_str.upper()))
                    processed_motion_events +=1
                except ValueError as e:
                    print(f"Warning: Could not parse datetime for line {line_count}: {line.strip()} - {e}")
            
            if line_count % 100000 == 0: # Optional: print progress for very large files
                 print(f"  ... processed {line_count} lines, found {processed_motion_events} motion events.")
    
    print(f"Finished processing {line_count} lines. Total motion events found: {processed_motion_events}.")
    
    if not raw_events:
        print("No motion sensor events were successfully parsed.")
        return raw_events

    print("Sorting motion events by timestamp...")
    raw_events.sort(key=lambda x: x[0]) # Sort by datetime
    print("Sorting complete.")
    return raw_events

def create_frames_from_events(raw_events, sensor_map, grid_height, grid_width):
    """
    Processes raw sensor events to create frames based on state changes.
    Each frame highlights sensors that changed state at a particular time.
    Returns a list of dictionaries: [{'frame': np.array, 'duration': float}]
    """
    if not raw_events:
        print("create_frames_from_events: No raw events to process.")
        return []

    sensor_last_states = {} # Stores the last known state ('ON' or 'OFF') of each sensor
    change_events = [] # Stores {'time_sec': float, 'sensor_id': str, 'state': str}

    # Determine the first timestamp from the already sorted raw_events
    first_timestamp = raw_events[0][0]
    print(f"First event timestamp: {first_timestamp}")

    for dt_obj, sensor_id, current_state_str in raw_events:
        time_sec = (dt_obj - first_timestamp).total_seconds()
        
        last_known_state = sensor_last_states.get(sensor_id, 'OFF') # Assume sensors are initially 'OFF'

        if current_state_str != last_known_state:
            # Record the change along with the new state
            change_events.append({'time_sec': time_sec, 'sensor_id': sensor_id, 'new_state': current_state_str})
            sensor_last_states[sensor_id] = current_state_str # Update the last known state
            
    if not change_events:
        print("No state changes detected among motion sensors.")
        return []
    
    # change_events are already sorted by time because raw_events were sorted.

    # Group changes by their exact timestamp
    events_by_time = {}
    for cevent in change_events:
        t = cevent['time_sec']
        if t not in events_by_time:
            events_by_time[t] = []
        # Store sensor_id for frame generation; new_state was used to detect change
        events_by_time[t].append(cevent['sensor_id']) 

    output_frames_data = []
    sorted_event_times = sorted(events_by_time.keys()) # Get unique timestamps of changes

    print(f"Found {len(sorted_event_times)} unique timestamps with state changes.")

    for i, current_time_sec in enumerate(sorted_event_times):
        sensors_changed_this_instant = events_by_time[current_time_sec]
        
        frame = np.zeros((grid_height, grid_width), dtype=np.uint8)
        active_sensor_in_frame = False
        for sensor_id in sensors_changed_this_instant:
            if sensor_id in sensor_map:
                r, c = sensor_map[sensor_id]
                if 0 <= r < grid_height and 0 <= c < grid_width:
                    frame[r, c] = 1 # Mark sensor as changed in this frame
                    active_sensor_in_frame = True
                else:
                    print(f"Warning: Sensor {sensor_id} coordinates ({r},{c}) are out of grid bounds ({grid_height}x{grid_width}).")
            else:
                # This case should be less common if SENSOR_COORDINATES is comprehensive
                # print(f"Debug: Sensor {sensor_id} changed state but not in SENSOR_COORDINATES map.")
                pass


        # Only create a frame if at least one *mapped* sensor changed state.
        if not active_sensor_in_frame : 
            # print(f"Debug: Skipping frame at {current_time_sec}s as no *mapped* sensors changed.")
            continue

        duration = 0.0
        if i + 1 < len(sorted_event_times):
            next_event_time_sec = sorted_event_times[i+1]
            duration = next_event_time_sec - current_time_sec
        else:
            # For the last change event, duration is until the end of the observation period (last raw event)
            last_raw_event_time_sec = (raw_events[-1][0] - first_timestamp).total_seconds()
            if current_time_sec < last_raw_event_time_sec:
                duration = last_raw_event_time_sec - current_time_sec
            else: 
                # This change is the last one, or very close to the last raw event.
                # Give it a default small duration if it's at the very end.
                duration = 1.0 # Default duration for the very last frame if no further events.
        
        if duration < 0: # Should ideally not happen with sorted times
             print(f"Warning: Negative duration ({duration}s) calculated at time {current_time_sec}s. Clamping to 0.01s.")
             duration = 0.01 # Use a very small positive duration
        
        # Ensure a minimal positive duration if calculated as zero, unless it's truly instantaneous
        # and there's another event immediately after (which the logic handles by duration to next event).
        # This is more for the very last frame or if rounding leads to zero.
        if duration <= 1e-6 : # If effectively zero
            if i == len(sorted_event_times) - 1: # and it's the last frame
                 duration = 0.01 # Fallback for the very last frame if its calculated duration is zero
            elif sorted_event_times[i+1] - current_time_sec > 1e-6: # next event is distinctly later
                pass # duration is correctly very small
            else: # next event is at same time (should be grouped) or also zero duration
                duration = 0.01


        output_frames_data.append({'frame': frame, 'duration': duration})
        
    return output_frames_data

def save_processed_data(frames_data, output_filepath):
    """
    Saves the processed frames and durations to a compressed .npz file.
    """
    if not frames_data:
        print("No data to save (frames_data is empty).")
        return

    frames_list = np.array([item['frame'] for item in frames_data])
    durations_list = np.array([item['duration'] for item in frames_data])
    
    np.savez_compressed(output_filepath, frames=frames_list, durations=durations_list)
    print(f"Processed data saved to {output_filepath}")
    print(f"Saved {len(frames_list)} frames.")

def load_processed_data(npz_filepath):
    """
    Loads processed frames and durations from an .npz file.
    Returns a list of (frame_array, duration) tuples.
    """
    if not os.path.exists(npz_filepath):
        print(f"Error: File not found at {npz_filepath}")
        return []
        
    try:
        data = np.load(npz_filepath)
        if 'frames' not in data or 'durations' not in data:
            print(f"Error: The file {npz_filepath} does not contain 'frames' or 'durations' arrays.")
            return []
        loaded_frames = data['frames']
        loaded_durations = data['durations']
    except Exception as e:
        print(f"Error loading data from {npz_filepath}: {e}")
        return []
    
    return list(zip(loaded_frames, loaded_durations))

def print_frame(frame_array, sensor_coords_map, grid_h, grid_w):
    """
    Prints a string representation of the frame, showing sensor IDs and their ON/OFF status.
    'frame_array' is the 2D numpy array for the current frame.
    'sensor_coords_map' is the SENSOR_COORDINATES dictionary.
    'grid_h' and 'grid_w' are the dimensions of the grid.
    """
    # Create a reverse mapping from (row, col) to sensor_id for easier lookup
    coord_to_sensor_map = {coords: sensor_id for sensor_id, coords in sensor_coords_map.items()}
    
    print_buffer = []
    header = "Frame Status:"
    print_buffer.append(header)
    print_buffer.append("-" * (grid_w * 12)) # Adjust multiplier for desired width

    for r in range(grid_h):
        row_str = "|"
        for c in range(grid_w):
            sensor_id_at_loc = coord_to_sensor_map.get((r, c), "----") # Get sensor ID or placeholder
            status_val = frame_array[r, c]
            status_str = "ON" if status_val == 1 else "off"
            
            # Format each cell to be roughly the same width for alignment
            # Example: "M001(ON)  " or "----(off) "
            cell_str = f"{sensor_id_at_loc: <4}({status_str: >3}) |" 
            row_str += f" {cell_str: <10}" # Adjust padding for cell width
        print_buffer.append(row_str)
    print_buffer.append("-" * (grid_w * 12))
    print("\n".join(print_buffer))


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure data.txt is in the same directory as the script, or provide the correct path.
    log_file = "data.txt" 
    output_npz_file = "processed_sensor_frames.npz"

    # 1. Load and filter log data
    print(f"Starting log processing for {log_file}...")
    raw_sensor_events = load_and_filter_log_data(log_file)
    
    if not raw_sensor_events:
        print("No motion sensor events found or file error. Exiting.")
    else:
        print(f"Successfully loaded and sorted {len(raw_sensor_events)} raw motion sensor events.")
        
        # 2. Create frames based on sensor state changes
        print("Creating frames from events...")
        frames_with_durations = create_frames_from_events(
            raw_sensor_events, 
            SENSOR_COORDINATES, 
            GRID_HEIGHT, 
            GRID_WIDTH
        )
        
        if not frames_with_durations:
            print("No frames generated (e.g., no state changes detected for mapped sensors, or other issues).")
        else:
            print(f"Generated {len(frames_with_durations)} frames with durations.")
            
            # 3. Save the processed data
            save_processed_data(frames_with_durations, output_npz_file)
            
            # 4. Example of loading the data back
            print(f"\nLoading data back from {output_npz_file} for verification...")
            loaded_data = load_processed_data(output_npz_file)
            
            if loaded_data:
                print(f"Successfully loaded {len(loaded_data)} (frame, duration) pairs.")
                
                # Example of using print_frame for the first few loaded frames
                print("\n--- Sample of Loaded Data with print_frame ---")
                for i, (frame, duration) in enumerate(loaded_data[:min(20, len(loaded_data))]): # Print first 3 frames
                    print(f"\nLoaded Frame {i}: duration = {duration:.3f}s")
                    print_frame(frame, SENSOR_COORDINATES, GRID_HEIGHT, GRID_WIDTH)
                    if i >= 19: # Limit to 3 frames for brevity in output
                        break 
                print("--- End of Sample ---")
            else:
                print("Failed to load data or no data was saved/found in the .npz file.")
    print("Processing complete.")
