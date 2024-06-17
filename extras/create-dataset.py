import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_captcha(text, font_path='path_to_font.ttf'):
    # Define the image size
    width, height = 200, 80
    background_color = (255, 255, 255)
    text_color = (0, 128, 0)
    line_color = (0, 128, 0)

    # Create a new image with white background
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)


    # Draw the large text on the image
    large_font = ImageFont.truetype(font_path, 50)
    text_width, text_height = draw.textbbox((0, 0), text[0], font=large_font)[2:]
    text_y = (height - text_height) // 2
    draw.text((10, text_y), text[0], font=large_font, fill=text_color)
    draw.text((71, text_y), text[2], font=large_font, fill=text_color)
    draw.text((130, text_y), text[4], font=large_font, fill=text_color)

    # Draw the small text on the image
    large_font = ImageFont.truetype(font_path, 50)
    text_width, text_height = draw.textbbox((0, 0), text[0], font=large_font)[2:]
    text_y = (height - text_height) // 2
    draw.text((40, text_y - 14), text[1], font=large_font, fill=text_color)
    draw.text((100, text_y - 14), text[3], font=large_font, fill=text_color)
    draw.text((160, text_y -14), text[5], font=large_font, fill=text_color)


    # Draw the 1st vertical line through the text
    # start_point = (0, height // 3)
    # end_point = (width, height // 3)
    # draw.line([start_point, end_point], fill=(254,249,243), width=1)

    # Draw the 2nd vertical line through the text
    # start_point = (0, height // 2)
    # end_point = (width, height // 2)
    # draw.line([start_point, end_point], fill=(183,212,230), width=1)

    # Draw the 3rd vertical line through the text
    # start_point = (0, height // 1.5)
    # end_point = (width, height // 1.5)
    # draw.line([start_point, end_point], fill=(214,166,208), width=1)


    # Draw the horizontal line through the text
    start_point = (0, height)
    end_point = (width, 0)
    draw.line([start_point, end_point], fill=line_color, width=1)

    return image


# Example usage
from nanoid import generate
from clean import clean_image
import cv2
font_path = "./nimbus-roman-no9-l.regular-italic.otf"  # Replace with the path to your font file

for i in range(500):
    captcha_text = generate('1234567890abcdefghijlkmnopqrstuvwxyz', 6)
    captcha_image = generate_captcha(captcha_text, font_path)
    captcha_image = clean_image(cv2.cvtColor(np.asarray(captcha_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"../data/synthetic/valid/{captcha_text}.png", captcha_image)  # Save the image

