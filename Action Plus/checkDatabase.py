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
    tables = ['movement_data', 'movement_data_archive', 'athletes']
    
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"\n{table}: {count} record(s)")
        
        if count > 0:
            # Get sample data
            if table == 'movement_data':
                c.execute("SELECT DISTINCT participant_name, test_date, movement_type FROM movement_data LIMIT 10")
                rows = c.fetchall()
                print("  Sample records:")
                for row in rows:
                    print(f"    - {row[0]}, {row[1]}, {row[2]}")
            
            elif table == 'movement_data_archive':
                c.execute("SELECT DISTINCT participant_name, test_date, movement_type FROM movement_data_archive LIMIT 10")
                rows = c.fetchall()
                print("  Sample records:")
                for row in rows:
                    print(f"    - {row[0]}, {row[1]}, {row[2]}")
            
            elif table == 'athletes':
                c.execute("SELECT participant_name, test_date, movement_type, num_movements, total_sessions FROM athletes")
                rows = c.fetchall()
                print("  All athlete records:")
                for row in rows:
                    print(f"    - {row[0]}, {row[1]}, {row[2]}: {row[3]} movements, {row[4]} sessions")
    
    conn.close()
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("\nTo archive and update athletes:")
    print("  - Run: python main.py (which does this automatically)")


if __name__ == "__main__":
    check_database_status()

