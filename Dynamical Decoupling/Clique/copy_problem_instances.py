#!/usr/bin/env python3
"""
Copy problem instances from Clique.db to BatchClique.db
"""

import sqlite3
from benchmarklib.quantum_trials import BenchmarkDatabase
from benchmarklib.clique import CliqueProblem, CliqueTrial

def copy_problem_instances():
    # Connect to source database
    source_conn = sqlite3.connect('Clique.db')
    
    # Create new database with proper schema
    target_db = BenchmarkDatabase('BatchClique.db', CliqueProblem, CliqueTrial)
    
    # Copy problem instances
    cursor = source_conn.cursor()
    cursor.execute("SELECT * FROM problem_instances")
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} problem instances to copy")
    
    # Insert into target database
    target_conn = sqlite3.connect('BatchClique.db')
    target_cursor = target_conn.cursor()
    
    # Build insert statement
    placeholders = ', '.join(['?' for _ in columns])
    insert_sql = f"INSERT INTO problem_instances ({', '.join(columns)}) VALUES ({placeholders})"
    
    target_cursor.executemany(insert_sql, rows)
    target_conn.commit()
    
    print(f"Copied {len(rows)} problem instances to BatchClique.db")
    
    # Verify copy
    target_cursor.execute("SELECT COUNT(*) FROM problem_instances")
    count = target_cursor.fetchone()[0]
    print(f"Verification: {count} instances in target database")
    
    source_conn.close()
    target_conn.close()

if __name__ == "__main__":
    copy_problem_instances()