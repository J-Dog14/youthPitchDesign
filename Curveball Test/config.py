"""
Configuration file containing all file paths and database settings.
"""

# Database path (absolute path to ensure consistency)
import os
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pitch_kinematics.db")

# Input file paths (hardcoded to hard drive location)
EVENTS_PATH = r"D:\Youth Pitch Design\Exports\events.txt"
LINK_MODEL_BASED_PATH = r"D:\Youth Pitch Design\Exports\link_model_based.txt"
ACCEL_DATA_PATH = r"D:\Youth Pitch Design\Exports\accel_data.txt"

# Reference file paths (hardcoded to hard drive location)
REF_EVENTS_PATH = r"D:\Youth Pitch Design\Exports\reference_events.txt"
REF_LINK_MODEL_BASED_PATH = r"D:\Youth Pitch Design\Exports\reference_link_model_based.txt"
REF_ACCEL_DATA_PATH = r"D:\Youth Pitch Design\Exports\reference_accel_data.txt"

# Report output settings
OUTPUT_DIR = r"D:\Youth Pitch Design\Reports\Curveball"
OUTPUT_DIR_TWO = r"G:\My Drive\Youth Pitch Reports\Reports\Curveball"
LOGO_PATH = r"C:\Users\q\PycharmProjects\Youth Pitch Design\8ctane - Faded 8 to Blue.png"

