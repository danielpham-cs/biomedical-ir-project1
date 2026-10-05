"""CSV export shared by the search page, Homework 2 pages and report scripts."""
import csv
import io


def rows_csv(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue()
