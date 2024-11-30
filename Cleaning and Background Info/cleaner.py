
import numpy as np
import pandas as pd

def gradestats(df):
    """
    This function prints some basic statistics about the dataframe-- total number of rows, number of unique students, number of graduates, and the percentage of graduates.
    """
    students = df["STUDENT"].nunique()
    grads = df.groupby("STUDENT")["DEG_CD"].apply(lambda x: x.notnull().any()).sum()
    percent = np.round( grads/students * 100,2)
    print('Rows:', len(df))
    print('Students:', students)
    print('Grads:', grads)
    print('Grad rate:', percent)

def print_column_summary(df):
    """
    This function prints a summary of the columns in the dataframe.
    """
    print("Column Summary:")
    print("{:<25} {:<50}".format('Column Name', 'Description'))
    print("-" * 75)
    print("{:<25} {:<50}".format('STUDENT', 'student identifier (integer)'))
    print("{:<25} {:<50}".format('MAJOR_CURR', 'major at time of taking the course'))
    print("{:<25} {:<50}".format('MAJOR_COLL', 'College of liberal arts for all courses'))
    print("{:<25} {:<50}".format('CLSFN_YR', 'Classification of the year (Freshman, Sophomore, etc.) when course was taken'))
    print("{:<25} {:<50}".format('ENTRY_SEM_CD', 'Semester of enrollment (\'F\' --> Fall, \'S\' --> Spring, \'1\' --> Summer)'))
    print("{:<25} {:<50}".format('ENTRY_CCYY', 'Year of enrollment'))
    print("{:<25} {:<50}".format('LEVEL', 'Irrelevant, to be dropped'))
    print("{:<25} {:<50}".format('SEM_CCYY', 'Year in which course was taken'))
    print("{:<25} {:<50}".format('SEM_CD', 'Semester in which course was taken'))
    print("{:<25} {:<50}".format('OFFER_DEPT_ABRVN', 'Same for all rows (Math department)'))
    print("{:<25} {:<50}".format('CRSE', 'Course number (3 digit numeric string)'))
    print("{:<25} {:<50}".format('SECT', 'Course section'))
    print("{:<25} {:<50}".format('GRADE_CATGORY', 'Values: \'C- OR BELOW\', \'ABOVE C-\', \'Satisfactory/Pass\''))
    print("{:<25} {:<50}".format('CRSE_TITLE', 'Course title'))
    print("{:<25} {:<50}".format('SEM_CCYY.1', 'Year of graduation; NaN if the student did not graduate'))
    print("{:<25} {:<50}".format('SEM_CD.1', 'Semester of graduation; NaN if the student did not graduate'))
    print("{:<25} {:<50}".format('DEG_CD', 'Abbreviation of degree earned; NaN if the student did not graduate'))
    print("{:<25} {:<50}".format('DEG_DATEDB', 'Date of degree; NaN if the student did not graduate'))
    print("{:<25} {:<50}".format('MAJOR_CURR.1', 'Major at time of graduation; NaN if the student did not graduate'))
    print("{:<25} {:<50}".format('DEG_DESCR', 'Detailed description of the degree, including honors, minors etc.'))
    print("{:<25} {:<50}".format('LAST_RGST_TERM', 'Last registered term'))
    print("{:<25} {:<50}".format('CLSFN_YR.1', 'Classification of year when degree was received; NaN if student did not graduate'))
    print("{:<25} {:<50}".format('MAJOR_CURR.2', 'Second major, if any'))
    print("{:<25} {:<50}".format('MAJOR_COLL.1', 'College from which degree was obtained'))
    print("{:<25} {:<50}".format('Workday Enrolled in Fall 2024', 'Workday Enrolled in Fall 2024'))
    print("{:<25} {:<50}".format('Workday Enrolled in Fall 2024 Class Standing', 'Workday Enrolled in Fall 2024 Class Standing'))
    print("{:<25} {:<50}".format('Workday Enrolled in Fall 2024 Primary Program of Study', 'Workday Enrolled in Fall 2024 Primary Program of Study'))
    print()

def drop_initial_cols(df):
    """
    This function drops the columns that are not needed for the analysis."""
    cols = ['STUDENT',
        'ENTRY_CCYY', 'ENTRY_SEM_CD', 
        'SEM_CCYY', 'SEM_CD', 
        'SEM_CCYY.1', 'SEM_CD.1', 'DEG_CD',
        'GRADE_CATGORY', 'CRSE', 'CRSE_TITLE']
    return df[cols]

def format_dates(df):
    """
    This function formats the year and date columns into a single column of floats. 
    """
    #subtract 2000 from 'ENTRY_CCYY', 'SEM_CCYY', and 'SEM_CCYY.1'
    for x in ['ENTRY_CCYY', 'SEM_CCYY', 'SEM_CCYY.1']:
        df[x] = df[x] - 2000

    #encode df['ENTRY_SEM_CD] as {'F':0.6, 'S':0.2, '1':0.4}
    #These floats represent how much of the (academic) year has passed at the time of enrollment.
    df['ENTRY_SEM_CD'] = df['ENTRY_SEM_CD'].replace({'F':0.6,
                                                    '1':0.4,
                                                    'S':0.0})

    #Next, we encode the semester in which the course was taken and the semester in which the student graduated
    for sem in ['SEM_CD', 'SEM_CD.1']:
        df[sem] = df[sem].replace({'F': 1.0, 
                                '1': 0.6, 
                                'S': 0.4})

    #create columns for entry, course, and grad semesters.
    df['ENT_SEM'] = df['ENTRY_CCYY'] + df['ENTRY_SEM_CD']
    df['CRSE_SEM'] = df['SEM_CCYY'] + df['SEM_CD']
    df['GRAD_SEM'] = df['SEM_CCYY.1'] + df['SEM_CD.1']
    return df

def remove_by_ENT_SEM(df):
    """
    This function removes rows corresponding to students that enrolled before 2008, or after Sring 2021.
    """
    
    #remove students who entered before 2008 (because the data is shaky)
    df = df[df['ENT_SEM'] >= 8]

    print('After removing students who entered before 2008:')
    gradestats(df)

    #remove students who entered on or after Spring 2021
    df = df[df['ENT_SEM'] < 21]

    print()
    print('After dropping students who enrolled in Spring 2021 or later:')
    gradestats(df)
    return df

def remove_graduate_degrees(df):
    """
    This function removes rows corresponding to students who earned a graduate degree.
    """
    #dropped degrees are graduate degrees. kept degrees are undergraduate degrees.
    dropped_degrees = ['MRE', 'MCP', 'MED', 'MFA', 'MTR', 'MHC', 'MFS', 'MHO', 'MAA', 'DVM', 'MBS', 'MBA', 'MA', 'MEN', 'MFN', 'MAT', 'PHD', 'MS']

    #get a list of all students who got a graduate degree
    grad_students = df[df['DEG_CD'].isin(dropped_degrees)]['STUDENT'].unique()

    #drop the above students
    df = df[~df['STUDENT'].isin(grad_students)]

    print('After dropping students who took classes towards a graduate degree')
    gradestats(df)
    return df

def remove_by_CRSE_SEM(df):
    """
    This function re-formats the CRSE_SEM and GRAD_SEM columns, and then removes rows corresponding to courses taken after 9.5 Semesters, and rows corresponding to courses taken after graduation.
    """
    #subtract ENT_SEM from CRSE_SEM and GRAD_SEM to get relative semester values
    df['CRSE_SEM'] = (df['CRSE_SEM'] - df['ENT_SEM']).apply(lambda x: np. round(x*2.5, 1))
    df['GRAD_SEM'] = (df['GRAD_SEM'] - df['ENT_SEM']).apply(lambda x: np. round(x*2.5, 1))

    #remove entries corresponding to courses taken after 3.8 years
    df = df[df['CRSE_SEM'] <= 9.5]

    print('After dropping courses taken after 9.5 semesters:')
    gradestats(df)

    #drop all rows for which CRSE_SEM > GRAD_SEM
    df = df[(df['GRAD_SEM'].isnull()) | (df['CRSE_SEM'] <= df['GRAD_SEM'])]

    print()
    print('After removing courses taken after graduation:')
    gradestats(df)
    return df

def remove_duplicates(df):
    """
    This function removes duplicate rows from the dataframe.
    """
    check = df[['STUDENT',
                'CRSE',
                'SEM_CCYY',
                'SEM_CD'
                ]]
    #check how many duplicate rows are there in the dataset
    print('Number of duplicate rows:', check.duplicated().sum())

    #get the indices of the duplicate rows and drop them from the 'df' dataset
    df.drop(index= check[check.duplicated()].index, inplace=True)

    print('After dropping duplicates:')
    gradestats(df)
    return df

def format_and_drop_CRSE(df):
    """
    This function formats the CRSE and CRSE TITLE columns, combines together some similar course, and drops courses with row enrollment.
    """
    #remove the 'X' and 'H' appearing in any CRSE name
    df['CRSE'] = df['CRSE'].str.replace('X', '')
    df['CRSE'] = df['CRSE'].str.replace('H', '')

    #remove all courses for which CRSE_TITLE contains 'EDUC'
    df = df[~df['CRSE_TITLE'].str.contains('EDUC')]

    #the courses x = '302' and '403' both refer to abstract algebra II, so let's combine them and call them '302'
    #these are supposed to be the same, so re-number the '403' to '302'
    df.loc[df['CRSE'] == '403', 'CRSE'] = '302'

    #the courses x = '142' and '145' are both variations of trig, so let's combine them and call them '142'
    df.loc[df['CRSE'] == '145', 'CRSE'] = '142'

    #the course x='407' is applied linear algebra; very similar to '317', so let's combine them and call them '317'
    df.loc[df['CRSE'] == '407', 'CRSE'] = '317'

    #the courses x='497' and '397' are both about teaching secondary school math, we combine them and call them '397'
    df.loc[df['CRSE'] == '497', 'CRSE'] = '397'

    #it's natural to combine x = '490' (Independent Study) with x='495' (Special topics) and call them '495'
    df.loc[df['CRSE'] == '490', 'CRSE'] = '495'

    #combine '106' (Discovering math) with '105' (Intro to math ideas) and call them '105'
    df.loc[df['CRSE'] == '106', 'CRSE'] = '105'

    #drop '181' (Life sci calc & MDL 1) because it is no longer offered
    df = df.drop(df[df['CRSE'] == '181'].index)

    #cast the course numbers as integers
    df['CRSE'] = df['CRSE'].astype(int)

    #replace all courses >= 500 with 500 (grad courses)
    df['CRSE'] = np.where(df['CRSE'] >= 500, 500, df['CRSE'])

    #for any row with CRSE = 500, replace the CRSE_TITLE with 'GRADUATE MATH'
    df['CRSE_TITLE'] = np.where(df['CRSE'] == 500, 'GRADUATE MATH', df['CRSE_TITLE'])

    #if any CRSE corresponds to more than one CRSE_TITLE, then replace all CRSE_TITLES with the first one that appears
    df['CRSE_TITLE'] = df.groupby('CRSE')['CRSE_TITLE'].transform('first')

    #if any CRSE_TITLE corresponds to more than one CRSE, then replace all CRSEs with the first one that appears
    df['CRSE'] = df.groupby('CRSE_TITLE')['CRSE'].transform('first')

    #remove courses with less than 100 students
    df = df.groupby('CRSE').filter(lambda x: len(x) >= 100)

    print('After formatting courses and titles, and removing courses with low enrollment:')
    gradestats(df)
    return df

def make_crse_counts_csv(df):
    """
    This function creates a csv file with the number of students who have taken each course; it returns a list of courses, sorted in descending order of number of students who have taken them.
    """
    #create a dataframe with the number of students who have taken each course
    df['CRSE'] = df['CRSE'].astype(str)
    crse_counts = df.groupby(by=['CRSE','CRSE_TITLE']).size().reset_index(name='COUNT').sort_values(by='COUNT', ascending=False).reset_index(drop=True)

    courses = crse_counts['CRSE'].tolist()
    #save the dataframe to a csv file
    crse_counts.to_csv('crse_counts.csv', index=False)

    print(f'Left with {len(courses)} courses:',courses)
    return df, courses

def make_CRSE_columns(df,courses):
    """
    This function formats the 'GRADE CATGORY' column and then creates a column for each course with the value 'GRADE CATGORY' * 'CRSE_SEM'.
    """
    #format the grades
    df['GRADE_CATGORY'] = df['GRADE_CATGORY'].map({'C- OR BELOW':-1,
                                                'ABOVE C-':1,
                                                'Satisfactory/Pass':1})
    #We add one column for each course and temporarily stick a 0 in each column
    for x in courses:
        df[x] = 0.0

    #Now, in each row, if a student has taken a course x, then the value in the column 'x' is replaced with the product of the 'GRADE_CATEGORY' and 'CRSE_SEM' 
    for x in courses:
        df.loc[df['CRSE'] == x, x] = df['GRADE_CATGORY'] * df['CRSE_SEM']

    return df

def agg_and_add_Y(df,final_cols):
    """
    This function aggregates the data by student, and then adds a column 'Y' to the dataframe, which is 1 if the student has graduated in 9.5 semesters, and 0 otherwise.
    """
    df = df[final_cols]

    #replace all null values with 0
    df = df.fillna(0)

    #group by student, choose the max value in each column
    df = df.groupby('STUDENT').agg(lambda x: x.loc[x.abs().idxmax()]).reset_index()

    #Add a column 'y' whose value is 0 if 'GRAD_SEM' is 0 or greater than 8, and 1 otherwise
    df['Y'] = 0
    df.loc[(df['GRAD_SEM'] > 0) & (df['GRAD_SEM'] <= 9.5), 'Y'] = 1

    print('After aggregating the dataset by student and adding the target variable:')
    print('Rows:', len(df))
    print('Students', df['STUDENT'].nunique())
    print('Grads:', df['Y'].sum())
    print('Grad rate:', np.round(df['Y'].mean() * 100, 2))
    return df
