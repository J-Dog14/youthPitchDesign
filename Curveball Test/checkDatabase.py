"""
Script to check database status and show what's in each table.
"""

import sqlite3
from config import DB_PATH


def check_database_status():
    """Display current database status."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("=" * 100)
    print("DATABASE STATUS CHECK")
    print("=" * 100)
    
    # Check all tables
    tables = ['pitch_data', 'reference_data', 'pitch_data_archive', 'athletes']
    
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"\n{table}: {count} record(s)")
        
        if count > 0:
            # Get sample data
            if table == 'pitch_data':
                c.execute("SELECT DISTINCT participant_name, pitch_date, pitch_type FROM pitch_data LIMIT 10")
                rows = c.fetchall()
                print("  Sample records:")
                for row in rows:
                    print(f"    - {row[0]}, {row[1]}, {row[2]}")
            
            elif table == 'reference_data':
                c.execute("SELECT filename, participant_name, pitch_date, pitch_type FROM reference_data")
                rows = c.fetchall()
                print("  All reference records:")
                for row in rows:
                    filename = row[0].split("\\")[-1] if "\\" in row[0] else row[0]
                    print(f"    - {filename}: {row[1]}, {row[2]}, {row[3]}")
            
            elif table == 'pitch_data_archive':
                c.execute("SELECT DISTINCT participant_name, pitch_date, pitch_type FROM pitch_data_archive LIMIT 10")
                rows = c.fetchall()
                print("  Sample records:")
                for row in rows:
                    print(f"    - {row[0]}, {row[1]}, {row[2]}")
            
            elif table == 'athletes':
                c.execute("SELECT participant_name, test_date, pitch_type, num_pitches, total_sessions FROM athletes")
                rows = c.fetchall()
                print("  All athlete records:")
                for row in rows:
                    print(f"    - {row[0]}, {row[1]}, {row[2]}: {row[3]} pitches, {row[4]} sessions")
    
    conn.close()
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("\nTo fix reference data:")
    print("  1. Check that REF_*_PATH files in config.py point to the correct reference files")
    print("  2. Or override paths at the top of updateReferenceData.py")
    print("  3. Run: python updateReferenceData.py")
    print("\nTo archive and update athletes:")
    print("  - Run: python manualArchive.py")
    print("  - Or run: python main.py (which does this automatically)")


if __name__ == "__main__":
    check_database_status()

