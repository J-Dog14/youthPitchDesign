"""
Database initialization and data ingestion functions.
"""

import sqlite3
from config import DB_PATH, CAPTURE_RATE
from parsers import parse_events_from_aPlus, parse_aplus_kinematics, parse_file_info
from utils import compute_score


def init_db():
    """
    Create the movement_data table if it doesn't exist.
    This table stores current/new data for report generation.
    Cleared at the start of each run.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    create_sql = """
    CREATE TABLE IF NOT EXISTS movement_data (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT UNIQUE,
      participant_name TEXT,
      test_date TEXT,
      movement_type TEXT,
      foot_contact_frame INTEGER,
      release_frame INTEGER,

      [Arm_Abduction@Footplant] REAL,
      [Max_Abduction] REAL,
      [Shoulder_Angle@Footplant] REAL,
      [Max_ER] REAL,
      [Arm_Velo] REAL,
      [Max_Torso_Rot_Velo] REAL,
      [Torso_Angle@Footplant] REAL,

      [Score] REAL
    );
    """
    c.execute(create_sql)
    
    # Create indexes for common queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_movement_data_participant_date ON movement_data(participant_name, test_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_movement_data_participant_type ON movement_data(participant_name, movement_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_movement_data_date ON movement_data(test_date)")
    
    conn.commit()
    conn.close()


def init_archive_db():
    """
    Create the movement_data_archive table if it doesn't exist.
    This table stores all historical movement data for long-term storage.
    Multiple instances of the same person/date are allowed.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    create_sql = """
    CREATE TABLE IF NOT EXISTS movement_data_archive (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        participant_name TEXT,
        test_date TEXT,
        movement_type TEXT,
        foot_contact_frame INTEGER,
        release_frame INTEGER,

        [Arm_Abduction@Footplant] REAL,
        [Max_Abduction] REAL,
        [Shoulder_Angle@Footplant] REAL,
        [Max_ER] REAL,
        [Arm_Velo] REAL,
        [Max_Torso_Rot_Velo] REAL,
        [Torso_Angle@Footplant] REAL,

        [Score] REAL,
        archive_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
    c.execute(create_sql)
    
    # Create indexes for common queries (no UNIQUE constraint - allows duplicates)
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_participant_date ON movement_data_archive(participant_name, test_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_participant_type ON movement_data_archive(participant_name, movement_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_date ON movement_data_archive(test_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_timestamp ON movement_data_archive(archive_timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_participant ON movement_data_archive(participant_name)")
    
    conn.commit()
    conn.close()


def clear_movement_data():
    """
    Clear all data from the movement_data table.
    
    This function is called at the start of each analysis run to ensure
    fresh data processing.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("DELETE FROM movement_data")
    conn.commit()
    
    c.execute("SELECT COUNT(*) FROM movement_data")
    count = c.fetchone()[0]
    
    conn.close()
    print(f"Cleared movement_data table. Remaining records: {count}")


def archive_movement_data():
    """
    Copy all data from movement_data table to movement_data_archive table.
    This preserves historical data even after movement_data is cleared for the next run.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if there's any data to archive
    c.execute("SELECT COUNT(*) FROM movement_data")
    count = c.fetchone()[0]
    
    if count == 0:
        conn.close()
        print("No new data to archive")
        return
    
    # Get all columns from movement_data (excluding id since archive has its own)
    c.execute("PRAGMA table_info(movement_data)")
    columns = [row[1] for row in c.fetchall() if row[1] != 'id']
    
    if not columns:
        conn.close()
        return
    
    # Build INSERT INTO ... SELECT statement
    # Wrap column names in square brackets to handle special characters like @
    columns_quoted = [f"[{col}]" for col in columns]
    columns_str = ", ".join(columns_quoted)
    insert_sql = f"""
    INSERT INTO movement_data_archive ({columns_str})
    SELECT {columns_str} FROM movement_data
    """
    
    c.execute(insert_sql)
    conn.commit()
    
    # Verify how many were archived
    c.execute("SELECT COUNT(*) FROM movement_data_archive")
    total_archived = c.fetchone()[0]
    
    conn.close()
    
    print(f"Archived {count} movement record(s) to movement_data_archive (total archived: {total_archived})")


def ingest_data(aPlusDataPath, aPlusEventsPath):
    """
    Ingest data into the movement_data table.
    
    Uses INSERT OR REPLACE so database builds over time without duplicates.
    
    Args:
        aPlusDataPath: Path to APlusData.txt file
        aPlusEventsPath: Path to aPlus_events.txt file
    """
    events_dict = parse_events_from_aPlus(aPlusEventsPath, capture_rate=CAPTURE_RATE)
    kinematics = parse_aplus_kinematics(aPlusDataPath)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    insert_sql = """
    INSERT OR REPLACE INTO movement_data (
      filename, participant_name, test_date, movement_type,
      foot_contact_frame, release_frame,

      [Arm_Abduction@Footplant],
      [Max_Abduction],
      [Shoulder_Angle@Footplant],
      [Max_ER],
      [Arm_Velo],
      [Max_Torso_Rot_Velo],
      [Torso_Angle@Footplant],

      [Score]
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    processed_count = 0
    for row in kinematics:
        fn = row.get("filename", "").strip()
        if not fn:
            continue
        fc = events_dict.get(fn, {}).get("foot_contact_frame")
        rel = events_dict.get(fn, {}).get("release_frame")
        p_name, p_date, m_type = parse_file_info(fn)

        # Pull the numeric fields from row
        abd_fp = row.get("Arm_Abduction@Footplant") or 0
        max_abd = row.get("Max_Abduction") or 0
        shld_fp = row.get("Shoulder_Angle@Footplant") or 0
        max_er = row.get("Max_ER") or 0
        arm_velo = row.get("Arm_Velo") or 0
        torso_velo = row.get("Max_Torso_Rot_Velo") or 0
        torso_ang = row.get("Torso_Angle@Footplant") or 0

        # Compute the score
        score_val = compute_score(
            arm_velo,
            torso_velo,
            abd_fp,
            shld_fp,
            max_er
        )

        vals = [
            fn,
            p_name,
            p_date,
            m_type,
            fc,
            rel,

            abd_fp,
            max_abd,
            shld_fp,
            max_er,
            arm_velo,
            torso_velo,
            torso_ang,

            score_val
        ]
        c.execute(insert_sql, vals)
        processed_count += 1

    conn.commit()
    conn.close()
    
    print(f"Processed {processed_count} movement record(s)")

