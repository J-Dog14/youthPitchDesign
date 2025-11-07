"""
Main execution script for youth pitch design analysis.
"""

import os
from config import EVENTS_PATH, DB_PATH
from database import (
    init_db, init_reference_db, init_archive_db,
    ingest_pitches_with_events, clear_pitch_data, archive_pitch_data
)
from athletes import init_athletes_db, update_athletes_summary
from parsers import parse_events
from reports import generate_curve_report


if __name__ == "__main__":
    print("=" * 80)
    print("YOUTH PITCH DESIGN ANALYSIS")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Database exists: {os.path.exists(DB_PATH)}")
    print()
    
    # Initialize all tables
    print("Initializing database tables...")
    init_db()
    init_reference_db()
    init_archive_db()
    init_athletes_db()
    print("Database tables initialized.\n")

    # Clear pitch_data table to ensure fresh data for this run
    # NOTE: Reference data is NOT cleared - it remains untouched for comparison
    print("Clearing pitch_data table for fresh run...")
    clear_pitch_data()

    # NOTE: Reference data is NOT modified here. It remains untouched and is only used
    # for comparison in report generation. To add/update reference data, use updateReferenceData.py

    # Parse events from the events file and ingest new pitch data
    print(f"\nParsing events from: {EVENTS_PATH}")
    events_dict = parse_events(EVENTS_PATH)
    print(f"Found {len(events_dict)} pitch(es) to process.\n")
    
    print("Ingesting pitch data into database...")
    ingest_pitches_with_events(events_dict)

    # Archive the processed data to the permanent archive table
    print("\nArchiving data to permanent storage...")
    archive_pitch_data()

    # Update athletes summary table with aggregated statistics
    # This tracks: name, date, number of pitches, number of sessions
    print("\nUpdating athletes summary table...")
    update_athletes_summary()

    # Now that the DB is populated, generate the report.
    # The report will use existing reference_data for comparison.
    print("\nGenerating PDF report...")
    generate_curve_report()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nDatabase location: {DB_PATH}")
    print("Use 'python checkDatabase.py' to view current database status.")

