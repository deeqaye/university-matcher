import sys
from pathlib import Path

# Calculate path to dias directory
# From: /e:/stuff/dome/dias/university-matcher/apps/universities/uni_find_wrapper.py
# To:   /e:/stuff/dome/dias/
current_file = Path(__file__).resolve()
# Go up: uni_find_wrapper.py -> universities -> apps -> university-matcher -> dias
dias_path = current_file.parent.parent.parent.parent

if str(dias_path) not in sys.path:
    sys.path.insert(0, str(dias_path))

from uni_find import preprocess_university_data, calculate_match_score

# Re-export functions for use in views
__all__ = ['preprocess_university_data', 'calculate_match_score']
