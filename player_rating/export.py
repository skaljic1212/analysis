"""
Export player ratings to Excel (Ratings + Events sheets) and CSV,
plus a printed summary table. Columns use the scouting vocabulary.
"""

import pandas as pd

from .rating import highlights, CATEGORY_COLUMNS


def build_table(stats):
    rows = []
    for pid, s in sorted(stats.items(), key=lambda kv: -kv[1]['rating']):
        c = s['categories']
        row = {
            'Player ID': pid,
            'Team': s['team'],
            'Role': s['role'],
            'Minutes': round(s['minutes'], 2),
            'Points': s['points'],
            'Points /10min': s['points_per_10min'],
            'Rating (0-10)': s['rating'],
            'Pass Accuracy %': (round(s['pass_accuracy'], 1)
                                if pd.notna(s['pass_accuracy']) else ''),
            'Challenge Win %': (round(s['challenge_win_rate'], 1)
                                if pd.notna(s['challenge_win_rate']) else ''),
        }
        for cat in CATEGORY_COLUMNS:
            row[cat] = c[cat]
        row.update({
            'Pace (sprints)': s['sprints'],
            'Distance (m)': s['distance_m'],
            'Distance /min': s['distance_per_min'],
            'Frames Tracked': s['frames'],
            'What They Did': highlights(s),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_events_table(events, fps=25):
    rows = []
    for e in events:
        grades = "; ".join(f"{pid}: {g:+.1f} ({cat})"
                           for pid, g, cat in e['credits'])
        rows.append({
            'Frame': e['frame'],
            'Time (s)': round(e['frame'] / fps, 1),
            'Category': e['category'],
            'Player': e.get('from', e.get('winner')),
            'Other Player': e.get('to', e.get('loser')),
            'Team': e.get('team', ''),
            'Grades': grades,
            'Detail': e['detail'],
        })
    return pd.DataFrame(rows)


def export_all(stats, events, fps=25,
               xlsx_path='output_videos/player_ratings.xlsx',
               csv_path='output_videos/player_ratings.csv'):
    table = build_table(stats)
    ev_table = build_events_table(events, fps=fps)

    table.to_csv(csv_path, index=False)
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as xl:
            table.to_excel(xl, sheet_name='Ratings', index=False)
            ev_table.to_excel(xl, sheet_name='Events', index=False)
            # readable column widths
            for sheet, df in (('Ratings', table), ('Events', ev_table)):
                ws = xl.sheets[sheet]
                for i, col in enumerate(df.columns, start=1):
                    width = max(len(str(col)),
                                df[col].astype(str).str.len().max() if len(df) else 0)
                    ws.column_dimensions[ws.cell(1, i).column_letter].width = \
                        min(60, width + 2)
        print(f"  Excel exported: {xlsx_path}")
    except Exception as e:
        print(f"  Excel export failed ({e}); CSV still written")
    print(f"  CSV exported:   {csv_path}")

    cols = ['Player ID', 'Team', 'Role', 'Points', 'Points /10min',
            'Rating (0-10)', 'Short pass', 'Long pass', 'Through ball',
            'Intercept', 'Challenge won', 'Challenge lost', 'Dribble',
            'Distance (m)']
    print(table[cols].to_string(index=False))
    return table
