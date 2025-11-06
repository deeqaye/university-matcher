import pandas as pd

def preprocess_university_data(df):
    # copying necessary columns
    df = df[['university', 'country', 'language', 'non_eng_level', 'IELTS_min',
             'GPA_min', 'SAT_min', 'international_cost_max', 'is_public']].copy()

    # encoding is_public
    df['is_public'] = df['is_public'].map({'Public': 1, 'Private': 0})

    # encoding non_eng_level
    df['non_eng_level'] = df['non_eng_level'].astype(str).str.strip().str.upper()

    # CEFR dictionary
    level_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6, '0': 0, 0: 0}
    df['non_eng_level_num'] = df['non_eng_level'].map(level_map).fillna(0).astype(float)

    # ensuring numeric columns
    numeric_cols = ['IELTS_min', 'GPA_min', 'SAT_min', 'international_cost_max']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return df

def calculate_match_score(df, user_input):
    def score_row(row):
        # hard constraint for public/private preference
        if (user_input['public_preference'] != -1 and 
            row['is_public'] != user_input['public_preference']):
            return 0
        
        # hard constraint for country
        if row['country'] not in user_input['countries']:
            return 0
        
        # hard constraint for language
        # If user knows ANY language that the university offers AND meets proficiency, it's a match
        uni_langs = [lang.strip().lower() for lang in row['language'].split(',')]
        lang_satisfied = False

        for i, lang in enumerate(user_input['languages']):
            if lang.lower() in uni_langs:
                # if English, use IELTS
                if lang.lower() == 'english':
                    if user_input['ielts'] >= row['IELTS_min']:
                        lang_satisfied = True
                        break  # Found a matching language, no need to check others
                else:
                    # if not English, use CEFR 
                    level_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6, '0': 0, 0: 0}
                    # Make sure we have a language level for this language
                    if i < len(user_input['lang_levels']):
                        user_level = user_input['lang_levels'][i]
                        user_level_num = level_map.get(user_level, 0)
                        if user_level_num >= row['non_eng_level_num']:
                            lang_satisfied = True
                            break  # Found a matching language, no need to check others
        
        # If no language matches, return 0
        if not lang_satisfied:
            return 0

        # hard constraints for GPA and SAT
        if user_input['gpa'] < row['GPA_min']:
            return 0
        if row['SAT_min'] > 0 and user_input['sat'] < row['SAT_min']:
            return 0

        # hard constraint for budget (allow ±5%)
        max_cost = row['international_cost_max']
        budget_ok = (
            (user_input['budget_max'] + 0.05*user_input['budget_max'] >= max_cost)
        )
        if not budget_ok:
            return 0

        return round(100)

    df['match_score'] = df.apply(score_row, axis=1)
    result = df[df['match_score'] == 100]
    
    # Sort by match score (all 100) - this keeps original CSV order
    # The views.py will handle country-based prioritization
    return result

if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    df = preprocess_university_data(data)

    user_input = {
        'countries': ['UK'],
        'languages': ['English'],
        'lang_levels': ['B2'],
        'ielts': 6.5,
        'gpa': 3.5,
        'sat': 1200,
        'budget_max': 80000,
        'public_preference': 1  # 1 for public, 0 for private, -1 for no preference
    }

    result = calculate_match_score(df, user_input)
    print(result[['university']])

