import sys
import pandas as pd
import pulp

def read_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                config[key.strip()] = value.strip()
    return config

def load_course_info(course_info_path):
    df = pd.read_csv(course_info_path)
    course_info = {}
    for _, row in df.iterrows():
        course_info[row["course_name"]] = {
            "session_length": row["session_length"],
            "num_sessions_per_week": row["num_sessions_per_week"],
            "is_large_class": bool(row["is_large_class"]),
            "10_percent_rule_exempted": bool(row["10_percent_rule_exempted"]),
            "is_a_TA_session": bool(row["is_a_TA_session"]),
            "must_on_days": row["mustOnDays"].split() if isinstance(row["mustOnDays"], str) else [],
            "start_time": row["start_time"],
            "end_time": row["end_time"]
        }
    return course_info

def load_course_schedule(schedule_path):
    df = pd.read_csv(schedule_path)
    course_schedule = {}
    for _, row in df.iterrows():
        course_schedule[row["course_name"]] = {
            "instructor_name": row["instructor_name"],
            "must_on_days": row["must_on_days"].split() if isinstance(row["must_on_days"], str) else [],
            "must_start_time": row["must_start_time"],
            "must_end_time": row["must_end_time"]
        }
    return course_schedule

def load_conflict_courses(conflict_path):
    conflicts = set()
    with open(conflict_path, "r") as f:
        for line in f:
            courses = line.strip().split()
            for i in range(len(courses)):
                for j in range(i + 1, len(courses)):
                    conflicts.add((courses[i], courses[j]))
    return conflicts

def load_instructor_prefs(pref_path):
    df = pd.read_csv(pref_path)
    instructor_prefs = {}
    for _, row in df.iterrows():
        instructor_prefs[row["instructor_name"]] = {
            "preferred_days": row["preferred_days"].split() if isinstance(row["preferred_days"], str) else [],
            "preferred_start_time": row["preferred_start_time"],
            "preferred_end_time": row["preferred_end_time"],
            "same_day_preference": bool(row["sameDay"])
        }
    return instructor_prefs

def define_ILP(course_info, course_schedule, conflict_courses, instructor_prefs):
    model = pulp.LpProblem("Course_Scheduling", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", [(c, t) for c in course_info for t in range(100)], cat="Binary")
    y = pulp.LpVariable.dicts("y", [(i, c) for i in instructor_prefs for c in course_schedule], cat="Binary")
    z = pulp.LpVariable.dicts("z", [c for c in course_schedule], cat="Binary")

    model += pulp.lpSum(z[c] for c in course_schedule), "Maximize_Preference_Satisfaction"

    # Constraints
    for c, info in course_info.items():
        # Course must be assigned exactly once
        model += pulp.lpSum(x[c, t] for t in range(100)) == 1, f"One_Slot_Per_Course_{c}"

        # Enforce required days
        if info["must_on_days"]:
            valid_times = [t for t in range(100) if any(day in info["must_on_days"] for day in ["M", "T", "W", "R", "F"])]
            model += pulp.lpSum(x[c, t] for t in valid_times) == info["num_sessions_per_week"], f"Required_Days_{c}"

        # Enforce required start and end times
        if info["start_time"] != "-":
            start_limit = int(info["start_time"].split(":")[0]) * 2
            model += pulp.lpSum(x[c, t] for t in range(start_limit)) == 0, f"Start_Time_{c}"
        if info["end_time"] != "-":
            end_limit = int(info["end_time"].split(":")[0]) * 2
            model += pulp.lpSum(x[c, t] for t in range(end_limit, 100)) == 0, f"End_Time_{c}"

    # No overlapping courses
    for c1, c2 in conflict_courses:
        for t in range(100):
            model += x[c1, t] + x[c2, t] <= 1, f"No_Overlap_{c1}_{c2}_{t}"

    # Instructor same-day constraint
    for instructor, courses in instructor_prefs.items():
        if courses["same_day_preference"]:
            for c1 in course_schedule:
                for c2 in course_schedule:
                    if c1 != c2 and course_schedule[c1]["instructor_name"] == instructor:
                        model += pulp.lpSum(x[c1, t] - x[c2, t] for t in range(100)) == 0, f"Same_Day_{c1}_{c2}"

    model.solve()

    return model, x

def write_output(model, x, output_schedule, output_heatmap):
    with open(output_schedule, "w") as f:
        f.write("course,time_slot\n")
        for (c, t), var in x.items():
            if pulp.value(var) == 1:
                f.write(f"{c},{t}\n")

    with open(output_heatmap, "w") as f:
        f.write("time_slot,num_courses\n")
        for t in range(100):
            num_courses = sum(1 for (c, time), var in x.items() if time == t and pulp.value(var) == 1)
            f.write(f"{t},{num_courses}\n")

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = read_config(config_path)

    course_info = load_course_info(config["CourseInfo"])
    course_schedule = load_course_schedule(config["CoursesThisQuarter"])
    conflict_courses = load_conflict_courses(config["ConflictCourses"])
    instructor_prefs = load_instructor_prefs(config["InstructorPref"])

    model, x = define_ILP(course_info, course_schedule, conflict_courses, instructor_prefs)
    write_output(model, x, "schedule.csv", "heatMap.txt")