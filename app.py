from flask import Flask, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np
import io
import base64
import requests
import os
import json

app = Flask(__name__)

FOREST_GREEN  = '#2A5C45'
TERRACOTTA    = '#C4714F'
WARM_CREAM    = '#F5EFE0'
DARK_GREEN    = '#1B3D2F'
SAGE_GREEN    = '#52A67A'
MUTED_GOLD    = '#D4A373'
SOFT_TERR     = '#E8A98A'
CHARCOAL      = '#2C2C2C'
MID_GREY      = '#666666'
LIGHT_GREY    = '#CCCCCC'

BAR_COLORS = [FOREST_GREEN, TERRACOTTA, SAGE_GREEN, MUTED_GOLD,
              SOFT_TERR, DARK_GREEN, '#A8C5B5', '#E8C9B0']

LOGO_URL = "https://onlyforteachers.co.uk/wp-content/uploads/2025/07/OFT-STACKED-COLOUR.png"

def fetch_logo():
    try:
        resp = requests.get(LOGO_URL, timeout=6)
        img = mpimg.imread(io.BytesIO(resp.content), format='png')
        return img
    except Exception:
        return None

def detect_likert_order(title):
    t = title.lower()
    if any(w in t for w in ['confident', 'confidence']):
        return ['Not at all confident', 'Slightly confident', 'Moderately confident', 'Very confident', 'Extremely confident']
    if any(w in t for w in ['adequate', 'adequacy', 'cpd', 'professional development']):
        return ['Very inadequate', 'Inadequate', 'Neither adequate nor inadequate', 'Adequate', 'Very adequate']
    if any(w in t for w in ['agree', 'disagree', 'balance', 'workload']):
        return ['Strongly disagree', 'Disagree', 'Neither agree nor disagree', 'Agree', 'Strongly agree']
    if any(w in t for w in ['likely', 'likelihood']):
        return ['Very unlikely', 'Unlikely', 'Neither likely nor unlikely', 'Likely', 'Very likely']
    if any(w in t for w in ['effective', 'effectiveness']):
        return ['Not at all effective', 'Slightly effective', 'Moderately effective', 'Very effective', 'Extremely effective']
    if any(w in t for w in ['satisf']):
        return ['Very dissatisfied', 'Dissatisfied', 'Neither satisfied nor dissatisfied', 'Satisfied', 'Very satisfied']
    if any(w in t for w in ['impact', 'affect']):
        return ['No impact', 'Minor impact', 'Moderate impact', 'Significant impact', 'Major impact']
    return None

def compute_counts(series, title):
    order = detect_likert_order(title)
    if order:
        counts = series.value_counts().reindex(order).dropna()
        return counts, True
    else:
        counts = series.value_counts().sort_values(ascending=True)
        return counts, False

def wrap_label(text, max_width=28):
    words = text.split()
    lines, current = [], []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_width and current:
            lines.append(' '.join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(' '.join(current))
    return '\n'.join(lines)

def plot_question(ax, series, title):
    total = len(series.dropna())
    counts, is_likert = compute_counts(series, title)

    if len(counts) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color=MID_GREY)
        ax.set_title(title, fontsize=9, color=DARK_GREEN, fontweight='bold')
        return

    n = len(counts)
    colors = BAR_COLORS[:n]

    if is_likert and n >= 4:
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'oft', [TERRACOTTA, WARM_CREAM, FOREST_GREEN], N=n)
        colors = [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]

    wrapped_labels = [wrap_label(str(label)) for label in counts.index]
    y_pos = range(n)

    bars = ax.barh(y_pos, counts.values, color=colors,
                   edgecolor='white', linewidth=0.6, height=0.65)

    for bar, val in zip(bars, counts.values):
        pct = val / total * 100 if total > 0 else 0
        label_x = bar.get_width() + max(counts.values) * 0.015
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{pct:.0f}%', va='center', ha='left',
                fontsize=8.5, color=CHARCOAL, fontweight='bold')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(wrapped_labels, fontsize=8)
    ax.set_xlim(0, max(counts.values) * 1.22)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=9,
                 color=DARK_GREEN, wrap=True, loc='left')

    for spine in ['bottom', 'right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(LIGHT_GREY)
    ax.spines['left'].set_linewidth(0.8)

    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_facecolor(WARM_CREAM)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.text(0.99, -0.06, f'n = {total}', transform=ax.transAxes,
            fontsize=7, ha='right', color=MID_GREY, style='italic')


@app.route('/generate-chart', methods=['POST'])
def generate_chart():
    try:
        payload = request.get_json(force=True)
        csv_data     = payload.get('csv_data', '')
        survey_title = payload.get('survey_title', 'UK Teacher Survey Results')
        columns      = payload.get('columns')
        skip_cols    = payload.get('skip_columns',
                           ['Timestamp', 'timestamp', 'id', 'ID',
                            'Email', 'email', 'Name', 'name', 'IP'])

        if not csv_data:
            return jsonify({'error': 'csv_data is required'}), 400

        df = pd.read_csv(io.StringIO(csv_data))

        if columns:
            cols_to_plot = [c for c in columns if c in df.columns]
        else:
            cols_to_plot = [c for c in df.columns
                            if not any(c.startswith(s) for s in skip_cols)][:9]

        n_charts = len(cols_to_plot)
        if n_charts == 0:
            return jsonify({'error': 'No plottable columns found'}), 400

        if n_charts <= 2:
            nrows, ncols_grid = 1, n_charts
        elif n_charts <= 4:
            nrows, ncols_grid = 2, 2
        elif n_charts <= 6:
            nrows, ncols_grid = 2, 3
        else:
            nrows, ncols_grid = 3, 3

        fig_width  = 6.5 * ncols_grid
        fig_height = 5.5 * nrows + 2.8

        fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white', dpi=150)

        header_bottom = 1 - (2.4 / fig_height)

        logo_img = fetch_logo()
        if logo_img is not None:
            logo_ax = fig.add_axes([0.02, header_bottom + 0.01, 0.10, 0.085])
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')

        fig.text(0.5, header_bottom + 0.068, survey_title,
                 ha='center', va='bottom', fontsize=20, fontweight='bold',
                 color=DARK_GREEN)
        fig.text(0.5, header_bottom + 0.038,
                 'Source: OnlyForTeachers - Original UK Teacher Survey Data  |  onlyforteachers.co.uk',
                 ha='center', va='bottom', fontsize=10, color=MID_GREY, style='italic')

        divider = plt.Line2D([0.04, 0.96], [header_bottom + 0.022, header_bottom + 0.022],
                             transform=fig.transFigure,
                             color=FOREST_GREEN, linewidth=2.5, solid_capstyle='round')
        fig.add_artist(divider)

        gs = GridSpec(nrows, ncols_grid, figure=fig,
                      top=header_bottom - 0.03,
                      bottom=0.07,
                      hspace=0.75, wspace=0.55)

        for i, col in enumerate(cols_to_plot):
            r, c = divmod(i, ncols_grid)
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor(WARM_CREAM)
            plot_question(ax, df[col], col)

        for i in range(n_charts, nrows * ncols_grid):
            r, c = divmod(i, ncols_grid)
            ax = fig.add_subplot(gs[r, c])
            ax.set_visible(False)

        fig.text(0.5, 0.025,
                 '\u00a9 OnlyForTeachers.co.uk  |  All survey data collected from registered UK teachers  |  '
                 'Cite as: "Source: OnlyForTeachers - Original UK Teacher Survey Data"',
                 ha='center', va='center', fontsize=8, color='#999999')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)

        chart_b64 = base64.b64encode(buf.read()).decode('utf-8')

        stats_summary = {}
        for col in cols_to_plot:
            total = len(df[col].dropna())
            counts, _ = compute_counts(df[col], col)
            stats_summary[col] = {
                'total_responses': total,
                'breakdown': {str(k): int(v) for k, v in counts.items()},
                'percentages': {str(k): round(int(v) / total * 100, 1)
                                for k, v in counts.items() if total > 0}
            }

        return jsonify({
            'success': True,
            'chart_base64': chart_b64,
            'columns_used': cols_to_plot,
            'stats_summary': stats_summary
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'OFT Chart Generator v2'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
