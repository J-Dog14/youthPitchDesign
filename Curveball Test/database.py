"""
Database initialization and data ingestion functions.
"""

import sqlite3
import pandas as pd
from config import DB_PATH, REF_EVENTS_PATH, REF_LINK_MODEL_BASED_PATH, REF_ACCEL_DATA_PATH, LINK_MODEL_BASED_PATH, ACCEL_DATA_PATH
from parsers import parse_events, parse_link_model_based_long, parse_accel_long
from utils import compute_pitch_stability_score, parse_file_info


def init_archive_db():
    """
    Create the pitch_data_archive table if it doesn't exist.
    This table stores all historical pitch data for long-term storage.
    Multiple instances of the same person/date are allowed.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    offsets = range(-20, 31)
    angle_cols = []
    for off in offsets:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
        angle_cols.extend([
            f"x_{lbl} REAL",
            f"y_{lbl} REAL",
            f"z_{lbl} REAL",
            f"ax_{lbl} REAL",
            f"ay_{lbl} REAL",
            f"az_{lbl} REAL"
        ])

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS pitch_data_archive (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        participant_name TEXT,
        pitch_date TEXT,
        pitch_type TEXT,
        foot_contact_frame INTEGER,
        release_frame INTEGER,
        pitch_stability_score REAL,
        archive_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        {", ".join(angle_cols)}
    )
    """
    c.execute(create_sql)
    
    # Create indexes for common queries (no UNIQUE constraint - allows duplicates)
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_participant_date ON pitch_data_archive(participant_name, pitch_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_participant_type ON pitch_data_archive(participant_name, pitch_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_date ON pitch_data_archive(pitch_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_timestamp ON pitch_data_archive(archive_timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_archive_participant ON pitch_data_archive(participant_name)")
    
    conn.commit()
    conn.close()


def archive_pitch_data():
    """
    Copy all data from pitch_data table to pitch_data_archive table.
    This preserves historical data even after pitch_data is cleared for the next run.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if there's any data to archive
    c.execute("SELECT COUNT(*) FROM pitch_data")
    count = c.fetchone()[0]
    
    if count == 0:
        conn.close()
        print("No new data to archive")
        return
    
    # Get all columns from pitch_data (excluding id since archive has its own)
    c.execute("PRAGMA table_info(pitch_data)")
    columns = [row[1] for row in c.fetchall() if row[1] != 'id']
    
    if not columns:
        conn.close()
        return
    
    # Build INSERT INTO ... SELECT statement
    columns_str = ", ".join(columns)
    insert_sql = f"""
    INSERT INTO pitch_data_archive ({columns_str})
    SELECT {columns_str} FROM pitch_data
    """
    
    c.execute(insert_sql)
    conn.commit()
    
    # Verify how many were archived
    c.execute("SELECT COUNT(*) FROM pitch_data_archive")
    total_archived = c.fetchone()[0]
    
    conn.close()
    
    print(f"Archived {count} pitch record(s) to pitch_data_archive (total archived: {total_archived})")


def clear_pitch_data():
    """
    Clear all data from the pitch_data table.
    
    NOTE: This does NOT affect reference_data, which remains untouched.
    This function is called at the start of each analysis run to ensure
    fresh data processing.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("DELETE FROM pitch_data")
    conn.commit()
    
    c.execute("SELECT COUNT(*) FROM pitch_data")
    count = c.fetchone()[0]
    
    conn.close()
    print(f"Cleared pitch_data table. Remaining records: {count}")


def init_db():
    """
    Create the pitch_data table if it doesn't exist.
    This table stores current/new data for report generation.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    offsets = range(-20, 31)
    angle_cols = []
    for off in offsets:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
        angle_cols.extend([
            f"x_{lbl} REAL",
            f"y_{lbl} REAL",
            f"z_{lbl} REAL",
            f"ax_{lbl} REAL",
            f"ay_{lbl} REAL",
            f"az_{lbl} REAL"
        ])

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS pitch_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        participant_name TEXT,
        pitch_date TEXT,
        pitch_type TEXT,
        foot_contact_frame INTEGER,
        release_frame INTEGER,
        pitch_stability_score REAL,
        {", ".join(angle_cols)}
    )
    """
    c.execute(create_sql)
    
    # Create indexes for common queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_pitch_data_participant_date ON pitch_data(participant_name, pitch_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pitch_data_participant_type ON pitch_data(participant_name, pitch_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pitch_data_date ON pitch_data(pitch_date)")
    
    conn.commit()
    conn.close()


def init_reference_db():
    """
    Create the reference_data table if it doesn't exist.
    This table stores reference/baseline data for comparison in reports.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    offsets = range(-20, 31)
    angle_cols = []
    for off in offsets:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
        angle_cols.extend([
            f"x_{lbl} REAL",
            f"y_{lbl} REAL",
            f"z_{lbl} REAL",
            f"ax_{lbl} REAL",
            f"ay_{lbl} REAL",
            f"az_{lbl} REAL"
        ])

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS reference_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        participant_name TEXT,
        pitch_date TEXT,
        pitch_type TEXT,
        foot_contact_frame INTEGER,
        release_frame INTEGER,
        pitch_stability_score REAL,
        {", ".join(angle_cols)}
    )
    """
    c.execute(create_sql)
    
    # Create indexes for common queries
    c.execute("CREATE INDEX IF NOT EXISTS idx_reference_data_pitch_type ON reference_data(pitch_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_reference_data_participant ON reference_data(participant_name)")
    
    conn.commit()
    conn.close()


def ingest_reference_data():
    """
    Ingest reference data into the reference_data table.
    
    NOTE: This function is only called by updateReferenceData.py.
    The main.py script does NOT modify reference_data to ensure it remains
    untouched for comparison purposes in reports.
    
    If a filename already exists, the record will be updated with new data.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    events_dict = parse_events(REF_EVENTS_PATH)
    df_angles = parse_link_model_based_long(REF_LINK_MODEL_BASED_PATH)
    df_accel = parse_accel_long(REF_ACCEL_DATA_PATH)
    df_merged = pd.merge(df_angles, df_accel, on="frame", how="inner")

    offsets = list(range(-20, 31))
    col_names = [
        "filename",
        "participant_name",
        "pitch_date",
        "pitch_type",
        "foot_contact_frame",
        "release_frame",
        "pitch_stability_score"
    ]
    for off in offsets:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
        col_names.extend([f"x_{lbl}", f"y_{lbl}", f"z_{lbl}", f"ax_{lbl}", f"ay_{lbl}", f"az_{lbl}"])

    placeholders = ",".join(["?"] * len(col_names))
    
    # Build update clause for all angle columns
    angle_cols_update = [col for col in col_names if col not in ["filename", "participant_name", "pitch_date", "pitch_type", "foot_contact_frame", "release_frame", "pitch_stability_score"]]
    update_clause = ", ".join([f"{col}=excluded.{col}" for col in angle_cols_update])
    
    insert_sql = f"""
    INSERT INTO reference_data ({",".join(col_names)})
    VALUES ({placeholders})
    ON CONFLICT(filename) DO UPDATE SET
        participant_name=excluded.participant_name,
        pitch_date=excluded.pitch_date,
        pitch_type=excluded.pitch_type,
        foot_contact_frame=excluded.foot_contact_frame,
        release_frame=excluded.release_frame,
        pitch_stability_score=excluded.pitch_stability_score,
        {update_clause}
    """

    print(f"Processing {len(events_dict)} reference pitches")
    inserted_count = 0
    updated_count = 0
    
    for pitch_idx, pitch_fp in enumerate(events_dict.keys()):
        foot_fr = events_dict[pitch_fp]["foot_contact_frame"]
        release_fr = events_dict[pitch_fp]["release_frame"]
        pitch_num = pitch_idx + 1

        x_col = f"x_p{pitch_num}"
        y_col = f"y_p{pitch_num}"
        z_col = f"z_p{pitch_num}"
        ax_col = f"ax_p{pitch_num}"
        ay_col = f"ay_p{pitch_num}"
        az_col = f"az_p{pitch_num}"

        if x_col not in df_merged.columns:
            print(f"WARNING: Skipping reference file {pitch_fp}, missing {x_col}")
            continue

        start_fr = release_fr - 20
        end_fr = release_fr + 30
        slice_df = df_merged[(df_merged["frame"] >= start_fr) & (df_merged["frame"] <= end_fr)]

        row_dict = {
            "filename": pitch_fp,
            "participant_name": parse_file_info(pitch_fp)[0],
            "pitch_date": parse_file_info(pitch_fp)[1],
            "pitch_type": parse_file_info(pitch_fp)[2],
            "foot_contact_frame": foot_fr,
            "release_frame": release_fr
        }

        for off in offsets:
            lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
            actual_fr = release_fr + off
            match = slice_df[slice_df["frame"] == actual_fr]
            if not match.empty:
                row_dict[f"x_{lbl}"] = match.iloc[0][x_col]
                row_dict[f"y_{lbl}"] = match.iloc[0][y_col]
                row_dict[f"z_{lbl}"] = match.iloc[0][z_col]
                row_dict[f"ax_{lbl}"] = match.iloc[0][ax_col]
                row_dict[f"ay_{lbl}"] = match.iloc[0][ay_col]
                row_dict[f"az_{lbl}"] = match.iloc[0][az_col]
            else:
                row_dict[f"x_{lbl}"] = None
                row_dict[f"y_{lbl}"] = None
                row_dict[f"z_{lbl}"] = None
                row_dict[f"ax_{lbl}"] = None
                row_dict[f"ay_{lbl}"] = None
                row_dict[f"az_{lbl}"] = None

        row_dict["pitch_stability_score"] = compute_pitch_stability_score(row_dict)
        values = [row_dict[col] for col in col_names]
        
        # Check if this filename already exists
        c.execute("SELECT COUNT(*) FROM reference_data WHERE filename = ?", (pitch_fp,))
        exists = c.fetchone()[0] > 0
        
        try:
            c.execute(insert_sql, values)
            if exists:
                updated_count += 1
                print(f"  Updated: {pitch_fp} ({row_dict['participant_name']}, {row_dict['pitch_date']}, {row_dict['pitch_type']})")
            else:
                inserted_count += 1
                print(f"  Inserted: {pitch_fp} ({row_dict['participant_name']}, {row_dict['pitch_date']}, {row_dict['pitch_type']})")
        except Exception as e:
            print(f"ERROR processing {pitch_fp}: {e}")
            import traceback
            traceback.print_exc()
            raise

    conn.commit()
    
    print(f"\nReference data ingestion complete:")
    print(f"  Inserted: {inserted_count} pitch(es)")
    print(f"  Updated: {updated_count} pitch(es)")
    
    conn.close()


def ingest_pitches_with_events(events_dict):
    """Ingest new pitch data into the pitch_data table using events_dict."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    df_angles = parse_link_model_based_long(LINK_MODEL_BASED_PATH)
    df_accel = parse_accel_long(ACCEL_DATA_PATH)
    df_merged = pd.merge(df_angles, df_accel, on="frame", how="inner", suffixes=("_ang", "_acc"))

    offsets = list(range(-20, 31))
    col_names = [
        "filename",
        "participant_name",
        "pitch_date",
        "pitch_type",
        "foot_contact_frame",
        "release_frame",
        "pitch_stability_score"
    ]
    for off in offsets:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
        col_names.extend([f"x_{lbl}", f"y_{lbl}", f"z_{lbl}", f"ax_{lbl}", f"ay_{lbl}", f"az_{lbl}"])

    placeholders = ",".join(["?"] * len(col_names))
    insert_sql = f"""
    INSERT INTO pitch_data ({",".join(col_names)})
    VALUES ({placeholders})
    ON CONFLICT(filename) DO UPDATE SET
        participant_name=excluded.participant_name,
        pitch_date=excluded.pitch_date,
        pitch_type=excluded.pitch_type,
        foot_contact_frame=excluded.foot_contact_frame,
        release_frame=excluded.release_frame,
        pitch_stability_score=excluded.pitch_stability_score
    """

    print(f"Processing {len(events_dict)} regular pitches")
    for pitch_idx, pitch_fp in enumerate(events_dict.keys()):
        foot_fr = events_dict[pitch_fp]["foot_contact_frame"]
        release_fr = events_dict[pitch_fp]["release_frame"]
        pitch_num = pitch_idx + 1

        x_col = f"x_p{pitch_num}"
        y_col = f"y_p{pitch_num}"
        z_col = f"z_p{pitch_num}"
        ax_col = f"ax_p{pitch_num}"
        ay_col = f"ay_p{pitch_num}"
        az_col = f"az_p{pitch_num}"

        if x_col not in df_merged.columns:
            print(f"WARNING: Skipping {pitch_fp}, missing {x_col}")
            continue

        start_fr = release_fr - 20
        end_fr = release_fr + 30
        slice_df = df_merged[(df_merged["frame"] >= start_fr) & (df_merged["frame"] <= end_fr)]

        row_dict = {
            "filename": pitch_fp,
            "participant_name": parse_file_info(pitch_fp)[0],
            "pitch_date": parse_file_info(pitch_fp)[1],
            "pitch_type": parse_file_info(pitch_fp)[2],
            "foot_contact_frame": foot_fr,
            "release_frame": release_fr
        }

        for off in offsets:
            lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
            actual_fr = release_fr + off
            match = slice_df[slice_df["frame"] == actual_fr]
            if not match.empty:
                row_dict[f"x_{lbl}"] = match.iloc[0][x_col]
                row_dict[f"y_{lbl}"] = match.iloc[0][y_col]
                row_dict[f"z_{lbl}"] = match.iloc[0][z_col]
                row_dict[f"ax_{lbl}"] = match.iloc[0][ax_col]
                row_dict[f"ay_{lbl}"] = match.iloc[0][ay_col]
                row_dict[f"az_{lbl}"] = match.iloc[0][az_col]
            else:
                row_dict[f"x_{lbl}"] = None
                row_dict[f"y_{lbl}"] = None
                row_dict[f"z_{lbl}"] = None
                row_dict[f"ax_{lbl}"] = None
                row_dict[f"ay_{lbl}"] = None
                row_dict[f"az_{lbl}"] = None

        row_dict["pitch_stability_score"] = compute_pitch_stability_score(row_dict)
        values = [row_dict[col] for col in col_names]
        c.execute(insert_sql, values)

    conn.commit()
    conn.close()

