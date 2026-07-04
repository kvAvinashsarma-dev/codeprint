def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate summary statistics for a list of numeric values.

    Args:
        values: A list of numeric values to analyze.

    Returns:
        A dictionary containing the mean, minimum, and maximum.

    Raises:
        ValueError: If the input list is empty.
    """
    if not values:
        raise ValueError("Cannot compute statistics for an empty list")
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def format_report(statistics: dict[str, float]) -> str:
    """Format the statistics dictionary as a human-readable report."""
    lines = [f"{name.capitalize()}: {value:.2f}"
             for name, value in statistics.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    sample_values = [23.5, 67.0, 45.2, 89.1, 12.8, 55.5]
    stats = calculate_statistics(sample_values)
    print(format_report(stats))
