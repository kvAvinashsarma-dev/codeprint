def get_average_grade(grades: list[float]) -> float:
    """Calculate the average of a list of grades.

    Args:
        grades: A list of numeric grade values.

    Returns:
        The arithmetic mean of the grades.

    Raises:
        ValueError: If the grades list is empty.
    """
    if not grades:
        raise ValueError("Cannot compute the average of an empty list")
    return sum(grades) / len(grades)


def determine_status(average: float, passing_threshold: float = 60.0) -> str:
    """Return 'pass' or 'fail' based on the average grade."""
    return "pass" if average >= passing_threshold else "fail"


if __name__ == "__main__":
    student_grades = [97, 79, 99, 40, 91]
    average_grade = get_average_grade(student_grades)
    print(f"Average grade: {average_grade:.2f}")
    print(f"Status: {determine_status(average_grade)}")
