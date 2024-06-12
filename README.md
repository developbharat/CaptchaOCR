# Captcha OCR
Most accurate Captcha and General OCR running on CPU and GPU

## Questions to Solve?
1. Why does the author set `n_input_length=12` when he is generating captchas of 4 characters in dataset? Does it effect model performance, or is it done during some experimentation and then code was never updated to change it? From my observation it seems like it was done on purpose, but i don't know why?
2. Is this implementation really accurate and can it work for general purpose ocr tasks? I don't know, will need to experiment with that?
3. Author is using 3 channels RGB images in the dataset and during model training, Does using Grayscale images reduce accuracy of model? Incase they don't then we must use Grayscale to even make model more performant!

## References
1. [https://github.com/ypwhs/captcha_break](https://github.com/ypwhs/captcha_break)
