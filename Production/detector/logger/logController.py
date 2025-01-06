# import csv
# import datetime
# import numpy as np
# from Production.detector.appState import AppState
#
# # Initialize logging
# def initialize_logging(filename='drowsiness_log.csv'):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Timestamp', 'Blink Count', 'Total Blinks', 'Average Blink Duration', 'Blink Rate (blinks/min)', 'Drowsy', 'Reasons'])
#
# # Function to log data
# def log_data(state: AppState, drowsy, reasons, filename='drowsiness_log.csv'):
#     current_time = datetime.datetime.now().isoformat()
#     blink_rate = state.update_blink_rate()
#     avg_blink_duration = np.mean(state.blink_durations[-10:]) if state.blink_durations else 0
#     with open(filename, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             current_time,
#             state.blink_counter,
#             state.total_blinks,
#             f"{avg_blink_duration:.3f}",
#             f"{blink_rate:.2f}",
#             drowsy,
#             "; ".join(reasons) if drowsy else ""
#         ])