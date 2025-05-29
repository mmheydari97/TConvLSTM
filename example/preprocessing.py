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
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

def load_and_filter_log_data(log_filepath, chunk_size=5000):
    """
    Reads the log file line by line, filters for motion sensors (M), 
    and parses relevant information.
    Returns a sorted list of events: (datetime_obj, sensor_id, state_str).
    """
    raw_events = []
    if not os.path.exists(log_filepath):
        print(f"Error: Log file not found at {log_filepath}")
        return raw_events

    print(f"Processing log file: {log_filepath} line by line...")
    line_count = 0
    processed_motion_events = 0
    with open(log_filepath, 'r') as f:
        for line in f: 
            line_count += 1
            parts = line.strip().split()
            if len(parts) < 4:
                continue 

            date_str, time_str, sensor_id, state_str = parts[0], parts[1], parts[2], parts[3]

            if sensor_id.startswith('M'): 
                try:
                    dt_obj = parse_datetime_flexible(date_str, time_str)
                    raw_events.append((dt_obj, sensor_id, state_str.upper()))
                    processed_motion_events +=1
                except ValueError as e:
                    print(f"Warning: Could not parse datetime for line {line_count}: {line.strip()} - {e}")
            
            if line_count % 100000 == 0: 
                 print(f"  ... processed {line_count} lines, found {processed_motion_events} motion events.")
    
    print(f"Finished processing {line_count} lines. Total motion events found: {processed_motion_events}.")
    
    if not raw_events:
        print("No motion sensor events were successfully parsed.")
        return raw_events

    print("Sorting motion events by timestamp...")
    raw_events.sort(key=lambda x: x[0]) 
    print("Sorting complete.")
    return raw_events

def create_frames_from_events(raw_events, sensor_map, grid_height, grid_width, representation_method="last_activated"):
    """
    Processes raw sensor events to create frames based on the specified representation method.
    Methods: "raw", "change_point", "last_activated" (default).
    Returns a list of dictionaries: [{'frame': np.array, 'duration': float}]
    """
    if not raw_events:
        print(f"create_frames_from_events (method: {representation_method}): No raw events to process.")
        return []

    print(f"Creating frames using '{representation_method}' representation method...")
    output_frames_data = []
    active_change_points_list = [] # List of (time_sec, grid_state)
    first_event_timestamp_obj = raw_events[0][0]

    if representation_method == "raw":
        # --- Raw Method Logic (Absolute ON/OFF state of all sensors) ---
        current_sensor_on_off_state_dict = {sid: 'OFF' for sid in sensor_map.keys()}
        current_master_grid_state = np.zeros((grid_height, grid_width), dtype=np.uint8)
        last_emitted_grid_state_for_deduplication = None

        for dt_obj, sensor_id, state_from_log_event in raw_events:
            time_sec_from_start = (dt_obj - first_event_timestamp_obj).total_seconds()
            if sensor_id not in sensor_map: continue

            previous_on_off_state_for_this_sensor = current_sensor_on_off_state_dict[sensor_id]
            if state_from_log_event != previous_on_off_state_for_this_sensor:
                current_sensor_on_off_state_dict[sensor_id] = state_from_log_event
                r, c = sensor_map[sensor_id]
                current_master_grid_state[r, c] = 1 if state_from_log_event == 'ON' else 0
                
                if last_emitted_grid_state_for_deduplication is None or \
                   not np.array_equal(current_master_grid_state, last_emitted_grid_state_for_deduplication):
                    active_change_points_list.append(
                        (time_sec_from_start, current_master_grid_state.copy())
                    )
                    last_emitted_grid_state_for_deduplication = current_master_grid_state.copy()
        
        if not active_change_points_list and raw_events: # If no effective changes, but events existed
            # current_master_grid_state holds the final state after all events
            active_change_points_list.append((0.0, current_master_grid_state.copy()))


    elif representation_method == "change_point":
        # --- Change Point Method Logic (Highlight sensors that just changed state) ---
        sensor_actual_states = {} # Tracks actual ON/OFF state to detect a change
        events_grouped_by_time_of_change = {}

        for dt_obj, sensor_id, current_state_from_log in raw_events:
            time_sec = (dt_obj - first_event_timestamp_obj).total_seconds()
            if sensor_id not in sensor_map: continue

            last_known_actual_state = sensor_actual_states.get(sensor_id, 'OFF')
            if current_state_from_log != last_known_actual_state:
                sensor_actual_states[sensor_id] = current_state_from_log # Update actual state
                if time_sec not in events_grouped_by_time_of_change:
                    events_grouped_by_time_of_change[time_sec] = []
                events_grouped_by_time_of_change[time_sec].append(sensor_id)
        
        sorted_event_times = sorted(events_grouped_by_time_of_change.keys())
        for t_sec in sorted_event_times:
            frame = np.zeros((grid_height, grid_width), dtype=np.uint8)
            sensors_changed_this_instant = events_grouped_by_time_of_change[t_sec]
            for sid in sensors_changed_this_instant:
                if sid in sensor_map: # Should always be true due to earlier check
                    r, c = sensor_map[sid]
                    frame[r, c] = 1 # Mark sensor as changed
            active_change_points_list.append((t_sec, frame))

        if not active_change_points_list and raw_events: # No changes detected
             active_change_points_list.append((0.0, np.zeros((grid_height, grid_width), dtype=np.uint8)))


    elif representation_method == "last_activated":
        # --- Last Activated Method Logic ---
        current_sensor_on_off_state_dict = {sid: 'OFF' for sid in sensor_map.keys()}
        _last_activated_sensor_id_on_grid = None # ID of sensor currently marked as 1
        
        # Add initial all-zeros frame at t=0 if the first activation is later
        # This ensures the timeline starts correctly.
        active_change_points_list.append((0.0, np.zeros((grid_height, grid_width), dtype=np.uint8)))

        for dt_obj, sensor_id, state_from_log_event in raw_events:
            time_sec = (dt_obj - first_event_timestamp_obj).total_seconds()
            if sensor_id not in sensor_map: continue

            actual_prev_on_off_state_of_sensor = current_sensor_on_off_state_dict[sensor_id]
            current_sensor_on_off_state_dict[sensor_id] = state_from_log_event # Update true ON/OFF state

            is_true_activation = (state_from_log_event == 'ON' and actual_prev_on_off_state_of_sensor == 'OFF')

            if is_true_activation:
                if sensor_id != _last_activated_sensor_id_on_grid: # A *new* sensor becomes "last activated"
                    _last_activated_sensor_id_on_grid = sensor_id
                    
                    new_la_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
                    r, c = sensor_map[_last_activated_sensor_id_on_grid]
                    new_la_grid[r, c] = 1
                    
                    # Check if this new grid is different from the last one added to the list
                    # (to handle multiple activations at the exact same timestamp if not pre-grouped)
                    # Or, more simply, if the _last_activated_sensor_id_on_grid actually changed.
                    # The current logic already ensures a change in _last_activated_sensor_id_on_grid.
                    
                    # If the new change point has the same time as the last one, update the last one.
                    if active_change_points_list and active_change_points_list[-1][0] == time_sec:
                        active_change_points_list[-1] = (time_sec, new_la_grid.copy())
                    else:
                        active_change_points_list.append((time_sec, new_la_grid.copy()))
        
        # Deduplicate consecutive identical frames that might arise from the initial 0.0 frame
        # if the first activation is also at 0.0
        if len(active_change_points_list) > 1 and active_change_points_list[0][0] == active_change_points_list[1][0] \
           and np.array_equal(active_change_points_list[0][1], active_change_points_list[1][1]):
            active_change_points_list.pop(0)
        
        # Ensure there's at least one frame if raw_events existed but no activations occurred
        # (initial 0.0 frame would cover this if no activations, otherwise first activation is covered)
        if not active_change_points_list and raw_events: # Should be covered by initial (0.0, zeros)
             active_change_points_list.append((0.0, np.zeros((grid_height, grid_width), dtype=np.uint8)))


    else:
        raise ValueError(f"Invalid representation_method: '{representation_method}'. Choose from 'raw', 'change_point', 'last_activated'.")

    # --- Common Duration Calculation and Final Frame List Generation ---
    if not active_change_points_list:
        print(f"Method '{representation_method}': No effective change points to form frames. Raw events count: {len(raw_events)}.")
        return []

    for i in range(len(active_change_points_list)):
        current_time, current_grid = active_change_points_list[i]
        
        duration = 0.0
        if i + 1 < len(active_change_points_list):
            next_time, _ = active_change_points_list[i+1]
            duration = next_time - current_time
        else: # Last frame in the sequence
            last_log_event_time_sec_from_start = (raw_events[-1][0] - first_event_timestamp_obj).total_seconds()
            if current_time < last_log_event_time_sec_from_start:
                duration = last_log_event_time_sec_from_start - current_time
            else: # current_time is at or beyond the last log event (e.g. if last change point is the last event)
                duration = 1.0 # Default duration for the very last state
        
        if duration <= 1e-6: # Ensure a minimal positive duration
            duration = 0.01 

        output_frames_data.append({'frame': current_grid, 'duration': duration})
        
    print(f"Method '{representation_method}': Generated {len(output_frames_data)} frames.")
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
    """
    coord_to_sensor_map = {coords: sensor_id for sensor_id, coords in sensor_coords_map.items()}
    
    print_buffer = []
    header = "Frame Status:"
    print_buffer.append(header)
    print_buffer.append("-" * (grid_w * 12)) 

    for r in range(grid_h):
        row_str = "|"
        for c in range(grid_w):
            sensor_id_at_loc = coord_to_sensor_map.get((r, c), "----") 
            status_val = frame_array[r, c] 
            status_str = "ON" if status_val == 1 else "off"
            
            cell_str = f"{sensor_id_at_loc: <4}({status_str: >3}) |" 
            row_str += f" {cell_str: <10}" 
        print_buffer.append(row_str)
    print_buffer.append("-" * (grid_w * 12))
    print("\n".join(print_buffer))


# --- Main Execution ---
if __name__ == "__main__":
    log_file = "data.txt" 
    
    print(f"Starting log processing for {log_file}...")
    raw_sensor_events = load_and_filter_log_data(log_file)
    
    if not raw_sensor_events:
        print("No motion sensor events found or file error. Exiting.")
    else:
        print(f"Successfully loaded and sorted {len(raw_sensor_events)} raw motion sensor events.")
        
        # --- Cycle through each representation method for demonstration ---
        # representation_methods_to_test = ["raw", "change_point", "last_activated"]
        representation_methods_to_test = ["last_activated"] # Default as per request, can test others

        for method in representation_methods_to_test:
            output_npz_file = f"processed_sensor_frames_{method}.npz"
            print(f"\n--- Processing with method: {method} ---")
            
            frames_with_durations = create_frames_from_events(
                raw_sensor_events, 
                SENSOR_COORDINATES, 
                GRID_HEIGHT, 
                GRID_WIDTH,
                representation_method=method
            )
            
            if not frames_with_durations:
                print(f"No frames generated for method '{method}'.")
            else:
                print(f"Method '{method}': Generated {len(frames_with_durations)} frames with durations.")
                save_processed_data(frames_with_durations, output_npz_file)
                
                print(f"\nLoading data back from {output_npz_file} for verification (method: {method})...")
                loaded_data = load_processed_data(output_npz_file)
                
                if loaded_data:
                    print(f"Successfully loaded {len(loaded_data)} (frame, duration) pairs for method '{method}'.")
                    
                    # print(f"\n--- Sample of Loaded Data with print_frame (method: {method}) ---")
                    # for i, (frame, duration) in enumerate(loaded_data[:min(5, len(loaded_data))]): # Print first 5 frames
                    #     print(f"\nLoaded Frame {i} (method: {method}): duration = {duration:.3f}s")
                    #     print_frame(frame, SENSOR_COORDINATES, GRID_HEIGHT, GRID_WIDTH)
                    #     if i >= 4: 
                    #         break 
                    print(f"--- End of Sample (method: {method}) ---")
                else:
                    print(f"Failed to load data or no data was saved/found for method '{method}'.")
            print(f"--- Finished processing method: {method} ---")
            
    print("\nAll processing complete.")

