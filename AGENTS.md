# AGENTS Instructions

This repository maintains a log of tasks performed by AI agents.

## Conventions
- Document each task under the History section.
- Run `python -m py_compile app.py` before committing to verify syntax.

## History
- Initial setup: added `requirements.txt` and `AGENTS.md`.
- Allow XLSX file uploads and added `openpyxl` dependency.

- Improve column detection and validation to handle missing `time` fields.
- Add sample dataset and prevent division by zero in spectral calculations.
- Remove sample XLSX dataset and document local data setup.
- Harden column mapping by normalizing headers and validating the time field.

- Guard analyses against missing wind columns.


