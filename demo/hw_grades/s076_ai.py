def calculate_average(grades: list[float]) -> float:
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


def determine_status(average: float, passing_threshold: float = 50.0) -> str:
    """Return 'pass' or 'fail' based on the average grade."""
    return "pass" if average >= passing_threshold else "fail"


if __name__ == "__main__":
    student_grades = [64, 80, 84, 49, 74]
    average_grade = calculate_average(student_grades)
    print(f"Average grade: {average_grade:.2f}")
    print(f"Status: {determine_status(average_grade)}")
