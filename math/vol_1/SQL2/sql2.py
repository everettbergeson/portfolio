# solutions.py
"""Volume 3: SQL 2.
<Name>
<Class>
<Date>
"""
import sqlite3 as sql
import numpy as np


# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # Query the database for the list of the names of students
    # who have a B grade in any course         
    results = cur.execute("SELECT SI.StudentName "
                "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "
                "ON SI.StudentID = SG.StudentID "
                "WHERE SG.Grade == 'B';").fetchall()
    conn.close()
    return np.ravel(results)

# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # Query the database for all tuples of the form (Name, MajorName, Grade) 
    results = cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                "ON SI.MajorID = MI.MajorID "           # 1. get major name
                "INNER JOIN StudentGrades AS SG "
                "ON SI.StudentID = SG.StudentID "       # 2. get grades
                "INNER JOIN CourseInfo AS CI "
                "ON CI.CourseID = SG.CourseID "         # 3. get course name
                "WHERE CI.CourseName == 'Calculus';").fetchall()
    cur.close()
    return results

# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        ((list): a list of strings, each of which is a course name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # Query the database for a list of course names
    results = cur.execute("SELECT CI.CourseName "
                "FROM CourseInfo as CI INNER JOIN StudentGrades AS SG "
                "ON CI.CourseID == SG.CourseID " 
                "GROUP BY SG.CourseID "             # Just keep those with at least 5
                "HAVING COUNT(*) >= 5;").fetchall()
    cur.close()
    return np.ravel(results)


# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    results = cur.execute("SELECT MI.MajorName, COUNT(*) as num_students "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                "ON SI.MajorID = MI.MajorID "
                "GROUP BY MI.MajorID "
                # Sort number of students ascending, then alphabetically by major
                "ORDER BY num_students ASC, MI.MajorName ASC;").fetchall()
    cur.close()
    return results


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # Get student name, major name
    results = cur.execute("SELECT SI.StudentName, MI.MajorName "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                "ON SI.MajorID = MI.MajorID "
                # Last name starts with C: '% C%'
                "WHERE SI.StudentName LIKE '% C%';").fetchall()
    cur.close()
    return results


# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    results = cur.execute("SELECT student_name, COUNT(*) AS num_classes, AVG(grade) AS gpa "           
        "FROM ( "           # Create sub-query to get student names and points
            "SELECT SI.StudentName AS student_name, CASE SG.Grade "
            "WHEN 'A+' THEN 4.0 "
            "WHEN 'A' THEN 4.0 "
            "WHEN 'A-' THEN 3.7 "
            "WHEN 'B+' THEN 3.4 "
            "WHEN 'B' THEN 3.0 "
            "WHEN 'B-' THEN 2.7 "
            "WHEN 'C+' THEN 2.4 "
            "WHEN 'C' THEN 2.0 "
            "WHEN 'C-' THEN 1.7 "
            "WHEN 'D+' THEN 1.4 "
            "WHEN 'D' THEN 1.0 "
            "WHEN 'D-' THEN 0.7 "
            "ELSE 0 END AS grade " #Link student id's to student names
            "FROM StudentInfo AS SI LEFT OUTER JOIN StudentGrades AS SG "
            "WHERE SI.StudentID = SG.StudentID) "
        "GROUP BY student_name "
        "ORDER BY gpa DESC;").fetchall()
    cur.close()
    return results



if __name__=="__main__":
    res = prob4()
    for r in res:
        print(r)