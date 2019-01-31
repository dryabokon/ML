from PIL import Image
import pytesseract
# ---------------------------------------------------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract/tesseract.exe'

# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    input_filename = 'data/ex18/input.jpg'
    print(pytesseract.image_to_string(Image.open(input_filename)))



