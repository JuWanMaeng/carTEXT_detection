import pytesseract

def textRead(image):
    # apply Tesseract v4 to OCR 
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    # display the text OCR'd by Tesseract
    print("OCR TEXT : {}\n".format(text))
    
    # strip out non-ASCII text 
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    print("Alpha numeric TEXT : {}\n".format(text))
    return text