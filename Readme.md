Questions about dataset:
- Can we get a key explaining what each column and value represents?
- Grades for each course?
- Can we organize dataset by student (replacing student ID with artificial numerical index)?

To-do:
- Ask our respective universities if they can provide similar datasets.
- In case we're not able to get our hands on grades data from respective university, need to formulate a projet topic using similar dataset (maybe College Scorecard or State-wide assessments?)

Project topic ideas:
- Classification problem: predict whether or not student will graduate in a 4-year period depending on their grades in certain math courses.
- 

Next meeting: Friday Oct 18 at 4:30 pm EST.


Notes from meeting on Oct 17:
- Caleb and Gonzalo will clean datasets
    - Add "y" column with 1 for graduated in 4 years and 0 for not graduated (in 4 years)
    - Count satisfactory/pass entries/ above C-/ C- or below
    - Clean typos in course names
    - If any student repeats a course, drop the earlier time they took it and leave the later time they took it
    - Grouped by entry year, compute the percentages of students that did graduate vs. did not graduate
    - Drop grad students/ grad courses
    - Given a course X, should we take into consideration that it may be required for many majors, or not (a kind of "course importance" column)? (One potential way to measure: get percentage of students in each major, get percentage of majors which require class X, take a weighted average to get percentage of students which were required to take X).
- Once cleaned, others will put together visualizations of any/all statistics we can think of.
- Obtain historical grade distributions of individual math courses?

Notes from meeting Oct 29
    - Treat all students who graduated in > 4 years as NG
    - Group together upper-division (proof-based) math courses (i.e. the ones taken only by math majors)
    - Remove topics courses (Independent study etc.)
    - Add data about student majors (first only, or extract all majors from the deg_descr column)
    - Drop classes taken after 4th year
    - Extract pre-reqs from catalog.ia.edu
    - Next meeting fixed for 7 pm EST Sunday Nov 3rd.
