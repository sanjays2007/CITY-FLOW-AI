"""
Database module for Smart Traffic Management System
Handles SQLite storage for detection history, signal phases, and traffic analytics
"""

import sqlite3
import os
from datetime import datetime, timedelta
from contextlib import contextmanager
import json

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'traffic_data.db')


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Detection records table - stores each detection cycle
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_ns INTEGER DEFAULT 0,
                vehicle_ew INTEGER DEFAULT 0,
                total_vehicles INTEGER DEFAULT 0,
                phase TEXT,
                green_time REAL,
                source_type TEXT,
                frame_number INTEGER
            )
        ''')
        
        # Signal phase history - tracks signal changes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                phase TEXT NOT NULL,
                duration REAL,
                vehicle_count_ns INTEGER,
                vehicle_count_ew INTEGER,
                load_status TEXT
            )
        ''')
        
        # Vehicle type counts - breakdown by vehicle type
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                car_count INTEGER DEFAULT 0,
                motorcycle_count INTEGER DEFAULT 0,
                bus_count INTEGER DEFAULT 0,
                truck_count INTEGER DEFAULT 0,
                other_count INTEGER DEFAULT 0
            )
        ''')
        
        # Daily summaries - aggregated daily statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_vehicles INTEGER DEFAULT 0,
                peak_hour INTEGER,
                peak_count INTEGER,
                avg_ns_count REAL,
                avg_ew_count REAL,
                total_ns INTEGER DEFAULT 0,
                total_ew INTEGER DEFAULT 0,
                ns_green_time_total REAL DEFAULT 0,
                ew_green_time_total REAL DEFAULT 0,
                congestion_events INTEGER DEFAULT 0
            )
        ''')
        
        # Hourly statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                hour INTEGER,
                total_vehicles INTEGER DEFAULT 0,
                ns_count INTEGER DEFAULT 0,
                ew_count INTEGER DEFAULT 0,
                avg_green_time REAL,
                UNIQUE(date, hour)
            )
        ''')
        
        conn.commit()
        print("[Database] Initialized successfully")


def insert_detection_record(vehicle_ns: int, vehicle_ew: int, phase: str, 
                           green_time: float, source_type: str = "webcam", 
                           frame_number: int = 0):
    """Insert a new detection record"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detection_records 
            (vehicle_ns, vehicle_ew, total_vehicles, phase, green_time, source_type, frame_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (vehicle_ns, vehicle_ew, vehicle_ns + vehicle_ew, phase, green_time, 
              source_type, frame_number))
        conn.commit()
        return cursor.lastrowid


def insert_signal_phase(phase: str, duration: float, vehicle_ns: int, 
                        vehicle_ew: int, load_status: str):
    """Insert a signal phase change record"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signal_history 
            (phase, duration, vehicle_count_ns, vehicle_count_ew, load_status)
            VALUES (?, ?, ?, ?, ?)
        ''', (phase, duration, vehicle_ns, vehicle_ew, load_status))
        conn.commit()
        return cursor.lastrowid


def insert_vehicle_types(car: int = 0, motorcycle: int = 0, bus: int = 0, 
                         truck: int = 0, other: int = 0):
    """Insert vehicle type breakdown"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO vehicle_types 
            (car_count, motorcycle_count, bus_count, truck_count, other_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (car, motorcycle, bus, truck, other))
        conn.commit()


def get_detection_history(limit: int = 100, offset: int = 0):
    """Get recent detection records"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM detection_records 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        return [dict(row) for row in cursor.fetchall()]


def get_signal_history(limit: int = 50):
    """Get recent signal phase history"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM signal_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_today_stats():
    """Get statistics for today"""
    today = datetime.now().strftime('%Y-%m-%d')
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_records,
                SUM(vehicle_ns) as total_ns,
                SUM(vehicle_ew) as total_ew,
                SUM(total_vehicles) as total_vehicles,
                AVG(vehicle_ns) as avg_ns,
                AVG(vehicle_ew) as avg_ew,
                MAX(total_vehicles) as peak_count
            FROM detection_records
            WHERE DATE(timestamp) = ?
        ''', (today,))
        result = cursor.fetchone()
        return dict(result) if result else None


def get_hourly_breakdown(date: str = None):
    """Get hourly breakdown for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                SUM(vehicle_ns) as ns_count,
                SUM(vehicle_ew) as ew_count,
                SUM(total_vehicles) as total,
                COUNT(*) as records
            FROM detection_records
            WHERE DATE(timestamp) = ?
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''', (date,))
        return [dict(row) for row in cursor.fetchall()]


def get_weekly_summary():
    """Get summary for the past 7 days"""
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                SUM(vehicle_ns) as total_ns,
                SUM(vehicle_ew) as total_ew,
                SUM(total_vehicles) as total_vehicles,
                AVG(vehicle_ns) as avg_ns,
                AVG(vehicle_ew) as avg_ew,
                COUNT(*) as records
            FROM detection_records
            WHERE DATE(timestamp) >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', (week_ago,))
        return [dict(row) for row in cursor.fetchall()]


def get_daily_summary(date: str = None):
    """Get detailed summary for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('''
            SELECT 
                SUM(vehicle_ns) as total_ns,
                SUM(vehicle_ew) as total_ew,
                SUM(total_vehicles) as total_vehicles,
                AVG(vehicle_ns) as avg_ns,
                AVG(vehicle_ew) as avg_ew,
                COUNT(*) as total_records,
                MIN(timestamp) as first_record,
                MAX(timestamp) as last_record
            FROM detection_records
            WHERE DATE(timestamp) = ?
        ''', (date,))
        basic_stats = dict(cursor.fetchone())
        
        # Peak hour
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                SUM(total_vehicles) as count
            FROM detection_records
            WHERE DATE(timestamp) = ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        ''', (date,))
        peak = cursor.fetchone()
        
        # Signal phase distribution
        cursor.execute('''
            SELECT 
                phase,
                COUNT(*) as count,
                SUM(duration) as total_duration
            FROM signal_history
            WHERE DATE(timestamp) = ?
            GROUP BY phase
        ''', (date,))
        phases = [dict(row) for row in cursor.fetchall()]
        
        return {
            'date': date,
            'stats': basic_stats,
            'peak_hour': dict(peak) if peak else None,
            'phase_distribution': phases
        }


def get_vehicle_type_stats(date: str = None):
    """Get vehicle type breakdown for a date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                SUM(car_count) as cars,
                SUM(motorcycle_count) as motorcycles,
                SUM(bus_count) as buses,
                SUM(truck_count) as trucks,
                SUM(other_count) as others
            FROM vehicle_types
            WHERE DATE(timestamp) = ?
        ''', (date,))
        result = cursor.fetchone()
        return dict(result) if result else None


def export_to_csv(start_date: str = None, end_date: str = None):
    """Export detection records to CSV format"""
    import csv
    import io
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, vehicle_ns, vehicle_ew, total_vehicles, 
                   phase, green_time, source_type
            FROM detection_records
            WHERE DATE(timestamp) BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (start_date, end_date))
        records = cursor.fetchall()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'NS Vehicles', 'EW Vehicles', 'Total', 
                     'Phase', 'Green Time (s)', 'Source'])
    
    for record in records:
        writer.writerow(list(record))
    
    return output.getvalue()


def generate_report_data(start_date: str = None, end_date: str = None):
    """Generate comprehensive report data for PDF export"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_records,
                SUM(vehicle_ns) as total_ns,
                SUM(vehicle_ew) as total_ew,
                SUM(total_vehicles) as total_vehicles,
                AVG(vehicle_ns) as avg_ns,
                AVG(vehicle_ew) as avg_ew,
                AVG(green_time) as avg_green_time
            FROM detection_records
            WHERE DATE(timestamp) BETWEEN ? AND ?
        ''', (start_date, end_date))
        overall = dict(cursor.fetchone())
        
        # Daily breakdown
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                SUM(total_vehicles) as vehicles,
                AVG(vehicle_ns) as avg_ns,
                AVG(vehicle_ew) as avg_ew
            FROM detection_records
            WHERE DATE(timestamp) BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_date, end_date))
        daily = [dict(row) for row in cursor.fetchall()]
        
        # Phase distribution
        cursor.execute('''
            SELECT 
                phase,
                COUNT(*) as count,
                AVG(duration) as avg_duration
            FROM signal_history
            WHERE DATE(timestamp) BETWEEN ? AND ?
            GROUP BY phase
        ''', (start_date, end_date))
        phases = [dict(row) for row in cursor.fetchall()]
        
        return {
            'period': {'start': start_date, 'end': end_date},
            'overall': overall,
            'daily_breakdown': daily,
            'phase_distribution': phases,
            'generated_at': datetime.now().isoformat()
        }


def update_hourly_stats():
    """Update hourly statistics table (call periodically)"""
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_hour = now.hour
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO hourly_stats (date, hour, total_vehicles, ns_count, ew_count, avg_green_time)
            SELECT 
                DATE(timestamp) as date,
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                SUM(total_vehicles),
                SUM(vehicle_ns),
                SUM(vehicle_ew),
                AVG(green_time)
            FROM detection_records
            WHERE DATE(timestamp) = ? AND CAST(strftime('%H', timestamp) AS INTEGER) = ?
            GROUP BY date, hour
        ''', (current_date, current_hour))
        conn.commit()


def cleanup_old_records(days_to_keep: int = 30):
    """Remove records older than specified days"""
    cutoff = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM detection_records WHERE DATE(timestamp) < ?', (cutoff,))
        cursor.execute('DELETE FROM signal_history WHERE DATE(timestamp) < ?', (cutoff,))
        cursor.execute('DELETE FROM vehicle_types WHERE DATE(timestamp) < ?', (cutoff,))
        conn.commit()
        print(f"[Database] Cleaned up records older than {cutoff}")


# Initialize database when module is imported
if __name__ == "__main__":
    init_database()
    print("Database initialized successfully!")
