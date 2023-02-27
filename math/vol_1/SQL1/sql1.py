# sql1.py
"""Volume 3: SQL 1 (Introduction).
Everett Bergeson
"""
import sqlite3 as sql
import  numpy as np
import csv
from  matplotlib import pyplot as plt

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    # Connect to database or create it if it doesn't exist
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Drop some tables if they exist
            cur.execute("DROP TABLE IF EXISTS MajorInfo")
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentGrades")

            # Add some tables
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT);")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT);")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER);")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT);")

            # Populate MajorInfo
            rows_major = [(1, 'Math'),(2, 'Science'),(3, 'Writing'),(4,'Art')]
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", rows_major)

            # Populate CourseInfo
            rows_course = [(1,'Calculus'),(2,'English'),(3,'Pottery'),(4,'History')]
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", rows_course)

            # Populate StudentInfo
            with open("student_info.csv", 'r') as infile:
                rows_info = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", rows_info)
            
            # Populate StudentGrades
            with open("student_grades.csv", 'r') as infile:
                rows_grades = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", rows_grades)
            
            # Replace -1 with NULL in StudentInfo table MajorID column
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1")
    finally:
        conn.close()

# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Drop old table
            cur.execute("DROP TABLE IF EXISTS USEarthquakes")
            # Create new table
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
            # Populate table
            with open("us_earthquakes.csv", 'r') as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)
            # Clean up table
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0")
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0")
    finally:
        conn.close()



# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # Query the database for all tuples of the form (StudentName, CourseName) 
    # where that student has an A or A+ grade in that course
    cur.execute("SELECT SI.StudentName, CI.CourseName "
                "FROM StudentInfo as SI, CourseInfo as CI, StudentGrades as SG "
                "WHERE SI.StudentID == SG.StudentID AND CI.CourseID == SG.CourseID "
                "AND (SG.Grade=='A' OR SG.Grade=='A+');")
    results = cur.fetchall()
    conn.close()
    return results


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # Query the database for:
    #   The magnitudes of the earthquakes during the 19th century
    #   The magnitudes of the earthquakes during the 20th century
    #   The average magnitude of all earthquakes in the database
    cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year < 1900 AND Year >= 1800")
    magnitudes_19th = np.ravel(cur.fetchall())
    cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year < 2000 AND Year >= 1900")
    magnitudes_20th = np.ravel(cur.fetchall())
    cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes")
    average = np.ravel(cur.fetchall())
    conn.close()
    
    # Create a histogram of the magnitude of the earthquakes from the 19th and 20th centuries
    ax1 = plt.subplot(121)
    ax1.hist(magnitudes_19th)
    ax1.set_xlabel("Magnitudes of Earthquakes")
    ax1.set_ylabel("Number of Earthquakes")
    ax1.set_title("19th Century Earthquakes")

    ax2 = plt.subplot(122)
    ax2.hist(magnitudes_20th)
    ax2.set_xlabel("Magnitudes of Earthquakes")
    ax2.set_ylabel("Number of Earthquakes")
    ax2.set_title("20th Century Earthquakes")

    plt.tight_layout()
    plt.show()

    return float(average)