"""
Main execution script for Action Plus movement analysis.
"""

import os
from config import DB_PATH, APLUS_EVENTS_PATH, APLUS_DATA_PATH
from database import (
    init_db, init_archive_db,
    ingest_data, clear_movement_data, archive_movement_data
)
from athletes import init_athletes_db, update_athletes_summary
from reports import generate_movement_report


if __name__ == "__main__":
    print("=" * 80)
    print("ACTION PLUS MOVEMENT ANALYSIS")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Database exists: {os.path.exists(DB_PATH)}")
    print()
    
    # Initialize all tables
    print("Initializing database tables...")
    init_db()
    init_archive_db()
    init_athletes_db()
    print("Database tables initialized.\n")
    
    # Clear movement_data table to ensure fresh data for this run
    print("Clearing movement_data table for fresh run...")
    clear_movement_data()
    
    # Ingest data
    print(f"\nReading data from:")
    print(f"  Events: {APLUS_EVENTS_PATH}")
    print(f"  Kinematics: {APLUS_DATA_PATH}\n")
    
    print("Ingesting movement data into database...")
    ingest_data(APLUS_DATA_PATH, APLUS_EVENTS_PATH)
    
    # Archive the processed data to the permanent archive table
    print("\nArchiving data to permanent storage...")
    archive_movement_data()
    
    # Update athletes summary table with aggregated statistics
    print("\nUpdating athletes summary table...")
    update_athletes_summary()
    
    # Generate report
    print("\nGenerating PDF report...")
    generate_movement_report()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nDatabase location: {DB_PATH}")
    print("Use 'python checkDatabase.py' to view current database status.")

