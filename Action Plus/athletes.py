"""
Athlete/Client summary table for tracking participant information and statistics.

This table aggregates data from movement_data_archive to provide quick access to:
- Participant name
- Test dates
- Number of movements per session
- Number of sessions
- Movement types tested
"""

import sqlite3
from config import DB_PATH


def init_athletes_db():
    """
    Create the athletes table if it doesn't exist.
    This table stores aggregated client/athlete information and statistics.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    create_sql = """
    CREATE TABLE IF NOT EXISTS athletes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_name TEXT NOT NULL,
        test_date TEXT NOT NULL,
        movement_type TEXT,
        num_movements INTEGER DEFAULT 0,
        avg_score REAL,
        min_score REAL,
        max_score REAL,
        first_session_date TEXT,
        last_session_date TEXT,
        total_sessions INTEGER DEFAULT 1,
        created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(participant_name, test_date, movement_type)
    )
    """
    c.execute(create_sql)
    
    # Create indexes for fast lookups
    c.execute("CREATE INDEX IF NOT EXISTS idx_athletes_name ON athletes(participant_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_athletes_date ON athletes(test_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_athletes_name_date ON athletes(participant_name, test_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_athletes_type ON athletes(movement_type)")
    
    conn.commit()
    conn.close()


def update_athletes_summary():
    """
    Update the athletes table with aggregated data from movement_data_archive.
    This creates/updates summary records for each participant/date/movement_type combination.
    Tracks: name, date, number of movements, number of sessions.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if archive table has data
    c.execute("SELECT COUNT(*) FROM movement_data_archive")
    archive_count = c.fetchone()[0]
    
    if archive_count == 0:
        conn.close()
        return
    
    # Aggregate data from archive - group by participant, date, and movement type
    # Count distinct archive_timestamp dates to get number of sessions
    summary_sql = """
    INSERT INTO athletes (
        participant_name, test_date, movement_type,
        num_movements, avg_score, min_score, max_score,
        first_session_date, last_session_date, total_sessions
    )
    SELECT 
        participant_name,
        test_date,
        movement_type,
        COUNT(*) AS num_movements,
        ROUND(AVG([Score]), 2) AS avg_score,
        ROUND(MIN([Score]), 2) AS min_score,
        ROUND(MAX([Score]), 2) AS max_score,
        MIN(archive_timestamp) AS first_session_date,
        MAX(archive_timestamp) AS last_session_date,
        COUNT(DISTINCT DATE(archive_timestamp)) AS total_sessions
    FROM movement_data_archive
    WHERE participant_name IS NOT NULL 
      AND test_date IS NOT NULL
      AND movement_type IS NOT NULL
    GROUP BY participant_name, test_date, movement_type
    ON CONFLICT(participant_name, test_date, movement_type) DO UPDATE SET
        num_movements = excluded.num_movements,
        avg_score = excluded.avg_score,
        min_score = excluded.min_score,
        max_score = excluded.max_score,
        last_session_date = excluded.last_session_date,
        total_sessions = excluded.total_sessions,
        updated_timestamp = CURRENT_TIMESTAMP
    """
    
    c.execute(summary_sql)
    conn.commit()
    
    # Count how many records were inserted/updated
    c.execute("SELECT COUNT(*) FROM athletes")
    total_athletes = c.fetchone()[0]
    
    conn.close()
    
    print(f"Updated athletes summary table: {total_athletes} record(s) total")


def get_athlete_summary(participant_name=None):
    """
    Retrieve athlete summary data.
    
    Args:
        participant_name: Optional filter by participant name
        
    Returns:
        list: List of tuples with (name, date, movement_type, num_movements, avg_score, total_sessions, last_session_date)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if participant_name:
        query = """
        SELECT participant_name, test_date, movement_type, num_movements, 
               avg_score, total_sessions, last_session_date
        FROM athletes
        WHERE participant_name = ?
        ORDER BY test_date DESC, movement_type
        """
        c.execute(query, (participant_name,))
    else:
        query = """
        SELECT participant_name, test_date, movement_type, num_movements, 
               avg_score, total_sessions, last_session_date
        FROM athletes
        ORDER BY participant_name, test_date DESC, movement_type
        """
        c.execute(query)
    
    results = c.fetchall()
    conn.close()
    
    return results

