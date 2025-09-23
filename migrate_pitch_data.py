import sqlite3
import os

DB_PATH = r"pitch_kinematics.db"  # adjust if needed

OFFSETS = list(range(-20, 31))

def new_schema_column_defs():
    cols = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "filename TEXT UNIQUE",
        "participant_name TEXT",
        "pitch_date TEXT",
        "pitch_type TEXT",
        "foot_contact_frame INTEGER",
        "release_frame INTEGER",
        "pitch_stability_score REAL",
    ]
    for off in OFFSETS:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"
        cols += [
            f"x_{lbl} REAL",
            f"y_{lbl} REAL",
            f"z_{lbl} REAL",
            f"ax_{lbl} REAL",
            f"ay_{lbl} REAL",
            f"az_{lbl} REAL",
        ]
    return cols

def ensure_backup(db_path):
    base, ext = os.path.splitext(db_path)
    backup_path = f"{base}.backup_before_migration{ext}"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    else:
        print(f"ℹ️ Backup already exists: {backup_path}")

def get_existing_columns(cur, table):
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

def column_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    for row in cur.fetchall():
        if row[1] == col:
            return True
    return False

def create_pitch_data_v2(cur):
    cols = new_schema_column_defs()
    cur.execute("DROP TABLE IF EXISTS pitch_data_v2")
    cur.execute(f"CREATE TABLE pitch_data_v2 ({', '.join(cols)})")
    print("✅ Created pitch_data_v2 with new schema.")

def build_insert_select_clause(cur):
    """
    Returns (insert_cols, select_exprs)
    insert_cols: list of column names for pitch_data_v2
    select_exprs: list of expressions selecting from old pitch_data (map what we can)
    """
    # v2 insert columns must match the new schema build
    insert_cols = ["filename", "participant_name", "pitch_date", "pitch_type",
                   "foot_contact_frame", "release_frame", "pitch_stability_score"]

    # y_*, z_* we can map from legacy if they exist:
    # - y_*  <- u_dev_*
    # - z_*  <- pron_*
    # x_* and all ax_*/ay_*/az_* will be NULL during migration

    # Figure out which legacy columns exist
    cur.execute("PRAGMA table_info(pitch_data)")
    old_cols = {row[1] for row in cur.fetchall()}

    # expressions to select from old table
    select_exprs = [
        "filename",
        "participant_name",
        "pitch_date",
        "pitch_type",
        "foot_contact_frame",
        "release_frame",
        "pitch_stability_score",
    ]

    # now all per-offset columns in v2
    for off in OFFSETS:
        lbl = f"neg{abs(off)}" if off < 0 else f"pos{off}"

        # x_*: no legacy mapping -> NULL
        insert_cols.append(f"x_{lbl}")
        select_exprs.append("NULL AS " + f"x_{lbl}")

        # y_* from u_dev_*
        insert_cols.append(f"y_{lbl}")
        legacy_y = f"u_dev_{lbl}"
        if legacy_y in old_cols:
            select_exprs.append(legacy_y + " AS " + f"y_{lbl}")
        else:
            select_exprs.append("NULL AS " + f"y_{lbl}")

        # z_* from pron_*
        insert_cols.append(f"z_{lbl}")
        legacy_z = f"pron_{lbl}"
        if legacy_z in old_cols:
            select_exprs.append(legacy_z + " AS " + f"z_{lbl}")
        else:
            select_exprs.append("NULL AS " + f"z_{lbl}")

        # ax_*/ay_*/az_*: no legacy per-axis mapping -> NULL
        for axis in ("ax", "ay", "az"):
            insert_cols.append(f"{axis}_{lbl}")
            select_exprs.append("NULL AS " + f"{axis}_{lbl}")

    return insert_cols, select_exprs

def migrate_data(conn):
    cur = conn.cursor()

    # sanity: old table must exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pitch_data'")
    if not cur.fetchone():
        raise RuntimeError("pitch_data table not found. Aborting migration.")

    # create target v2 table
    create_pitch_data_v2(cur)

    insert_cols, select_exprs = build_insert_select_clause(cur)

    sql = f"""
        INSERT INTO pitch_data_v2 ({", ".join(insert_cols)})
        SELECT {", ".join(select_exprs)}
        FROM pitch_data
    """
    cur.execute(sql)
    print(f"✅ Migrated {cur.rowcount if hasattr(cur, 'rowcount') else 'rows'} rows to pitch_data_v2.")

def swap_tables(conn):
    cur = conn.cursor()
    # keep a temp name to enable rollback steps if needed
    cur.execute("ALTER TABLE pitch_data RENAME TO pitch_data_old")
    cur.execute("ALTER TABLE pitch_data_v2 RENAME TO pitch_data")
    print("✅ Swapped tables: pitch_data_v2 → pitch_data")

    # Copy indexes/uniques if any were on old table and are still needed.
    # We at least ensure UNIQUE(filename) by recreating it explicitly:
    cur.execute("PRAGMA index_list(pitch_data)")
    existing_indexes = [row[1] for row in cur.fetchall()]  # index names
    # If not present, create one:
    if "idx_pitch_data_filename_unique" not in existing_indexes:
        try:
            cur.execute("CREATE UNIQUE INDEX idx_pitch_data_filename_unique ON pitch_data(filename)")
            print("✅ (Re)created unique index on filename.")
        except sqlite3.OperationalError:
            # Ignore if schema already enforces UNIQUE on column definition
            pass

    # Drop old table (AFTER successful swap)
    cur.execute("DROP TABLE IF EXISTS pitch_data_old")
    print("🧹 Dropped legacy pitch_data.")

def main():
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"DB not found at: {DB_PATH}")

    ensure_backup(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")  # lock DB for a safe migration

        migrate_data(conn)
        swap_tables(conn)

        conn.commit()
        print("🎉 Migration complete.")
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Optional: VACUUM (run in a separate connection)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()
        print("🧽 VACUUM complete.")

if __name__ == "__main__":
    main()
