from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO


def generate_pdf(email_text: str):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    x_margin = 40
    y_position = height - 40

    for line in email_text.split("\n"):
        if y_position < 40:
            pdf.showPage()
            y_position = height - 40

        pdf.drawString(x_margin, y_position, line)
        y_position -= 14

    pdf.save()
    buffer.seek(0)
    return buffer
