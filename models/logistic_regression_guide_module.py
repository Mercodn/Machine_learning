"""
Logistic Regression Guide Module
Generates a professional JPG summary image with uses and step-by-step case workflow.
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def get_logistic_regression_guide_image():
    """Generate a JPG image containing a logistic regression guide."""
    import numpy as np

    fig = plt.figure(figsize=(14, 9), facecolor='#10203f')
    ax = fig.add_subplot(111)
    ax.axis('off')

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('Blues'), extent=[0, 1, 0, 1], alpha=0.18)

    # Main title banner
    title_box = FancyBboxPatch(
        (0.03, 0.82), 0.94, 0.14,
        boxstyle='round,pad=0.03',
        linewidth=0,
        facecolor='#1c7ed6',
        edgecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(title_box)
    fig.text(0.05, 0.88, 'Logistic Regression Guide', fontsize=38, fontweight='bold', color='#ffffff', family='sans-serif')
    fig.text(0.05, 0.84, 'Colorful, readable summary of uses and a step-by-step case workflow', fontsize=16, color='#e7f6ff', family='sans-serif')

    badge = FancyBboxPatch(
        (0.74, 0.84), 0.20, 0.08,
        boxstyle='round,pad=0.03',
        linewidth=0,
        facecolor='#ffd43b',
        edgecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(badge)
    fig.text(0.76, 0.86, 'Max Score: 5.0', fontsize=14, fontweight='bold', color='#10203f', family='sans-serif')

    # Left content panel
    left_panel = FancyBboxPatch(
        (0.03, 0.48), 0.45, 0.30,
        boxstyle='round,pad=0.03',
        linewidth=0,
        facecolor='#eef6ff',
        edgecolor='#74c0fc',
        transform=ax.transAxes
    )
    ax.add_patch(left_panel)
    fig.text(0.05, 0.74, 'Key Uses of Logistic Regression', fontsize=22, fontweight='bold', color='#10203f', family='sans-serif')

    uses = [
        'Predict purchase intent and improve conversions',
        'Detect spam or fraudulent actions',
        'Support medical diagnosis with probability estimates',
        'Model credit risk and approval decisions',
        'Forecast customer churn and retention likelihood'
    ]
    for i, item in enumerate(uses):
        fig.text(0.05, 0.70 - i * 0.055, f'{i + 1}. {item}', fontsize=14, color='#283845', family='sans-serif')

    # Right content panel
    right_panel = FancyBboxPatch(
        (0.52, 0.48), 0.45, 0.30,
        boxstyle='round,pad=0.03',
        linewidth=0,
        facecolor='#fff7db',
        edgecolor='#ffd43b',
        transform=ax.transAxes
    )
    ax.add_patch(right_panel)
    fig.text(0.54, 0.74, 'Practical Step-by-Step Workflow', fontsize=22, fontweight='bold', color='#10203f', family='sans-serif')

    steps = [
        'Gather labeled examples with features and binary outcomes.',
        'Clean the data and prepare numeric features.',
        'Train logistic regression on the training set.',
        'Compute purchase probabilities for new inputs.',
        'Use a threshold (usually 0.5) to assign yes/no.'
    ]
    for i, step in enumerate(steps):
        fig.text(0.54, 0.70 - i * 0.055, f'{i + 1}. {step}', fontsize=14, color='#283845', family='sans-serif')

    # Bottom case panel
    bottom_panel = FancyBboxPatch(
        (0.03, 0.08), 0.94, 0.30,
        boxstyle='round,pad=0.03',
        linewidth=0,
        facecolor='#d8f5ff',
        edgecolor='#74c0fc',
        transform=ax.transAxes
    )
    ax.add_patch(bottom_panel)
    fig.text(0.05, 0.39, 'Case Study: Purchase Intent Prediction', fontsize=24, fontweight='bold', color='#10203f', family='sans-serif')

    example_lines = [
        'Goal: Predict if a website visitor will complete a purchase.',
        'Inputs: age, monthly income, visits, time on site, previous buys, discount usage.',
        'Output: probability of purchase and final yes/no decision.',
        'Impact: personalize offers, improve conversion rates, and allocate marketing budget.',
        'Outcome: readable logistic regression workflow ideal for teaching and screenshot capture.'
    ]
    for i, line in enumerate(example_lines):
        fig.text(0.05, 0.34 - i * 0.04, line, fontsize=14, color='#1f2a44', family='sans-serif')

    fig.text(0.05, 0.10, 'Inspired by bright educational menus and clear infographic layout.', fontsize=12, color='#10203f', family='sans-serif')
    fig.text(0.05, 0.06, 'More space, better contrast, and a design that reads easily on screen.', fontsize=12, color='#10203f', family='sans-serif')

    buffer = io.BytesIO()
    fig.savefig(buffer, format='jpeg', dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return f'data:image/jpeg;base64,{image_base64}'
