import pytesseract
from PIL import Image

content_list =[]

def extract_text_from_image(image_path, language='Sinhala'):
    # Open the image file
    with Image.open(image_path) as img:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(img, lang=language)
    return text

def content_list_from_frames(sorted_timestamps_list,language):
    print("OCR file - Language of the video ::: " + language)
    language_code = ""
    if language == "sinhala":
        language_code = "Sinhala"
    elif language == "english":
        language_code = "eng"
    if language_code != "":
        content_list = []
        for item in sorted_timestamps_list:
            image_path = "object_detection/test_images/timestamp{}.jpg".format(item)
            extracted_text = extract_text_from_image(image_path, language=language_code)
            content_before_newline = extracted_text.split('\n', 1)[0]
            if content_before_newline:
                content_list.append(content_before_newline)
            else:
                content_list.append("NA")
    else:
        print("No Valid Language Passed to OCR file")
    print("Extracted Texts List:")
    print(content_list)
    return content_list
