# utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
import os

# -------------------------------
# Function: Convert matplotlib figure to base64 string
# -------------------------------
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

# -------------------------------
# Function: Create a simple bar chart
# -------------------------------
def create_bar_chart(data, x_col, y_col, title="Bar Chart"):
    plt.figure(figsize=(6,4))
    sns.barplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xticks(rotation=45)
    fig = plt.gcf()
    img_base64 = plot_to_base64(fig)
    plt.close()
    return img_base64

# -------------------------------
# Function: Generate PDF report
# -------------------------------
def generate_pdf_report(filename, title, content_lines):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height - 50, title)

    # Content
    c.setFont("Helvetica", 12)
    y = height - 100
    for line in content_lines:
        c.drawString(50, y, line)
        y -= 20

    c.save()
    print(f"PDF report saved as: {filename}")

# -------------------------------
# Dummy function for generate_report
# -------------------------------
def generate_report(*args, **kwargs):
    """
    Temporary function to allow app.py import to work.
    You can replace with actual logic later.
    """
    print("generate_report called")
    return "report.pdf"

# -------------------------------
# Dummy function for get_dashboard_metrics
# -------------------------------
def get_dashboard_metrics(*args, **kwargs):
    """
    Temporary function to allow app.py import to work.
    You can replace with actual logic later.
    """
    print("get_dashboard_metrics called")
    return {
        "total_customers": 1000,
        "churned_customers": 200,
        "active_customers": 800
    }
