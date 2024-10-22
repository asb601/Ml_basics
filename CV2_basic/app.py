from PIL import Image
import cv2 
import matplotlib.pyplot as plt

image = cv2.imread("./WhatsApp Image 2024-10-14 at 16.26.03.jpeg")
#convert from BGR to RGB


if image is None :
    print("Couldnt load the image ")
else :
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Step 3: Display the original image
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')  # Hide axes for a clean display
    plt.show()
    
    pil_image = Image.fromarray(image_rgb)

    # Step 4: Crop the image using Pillow (left, top, right, bottom)
    # You can adjust these coordinates based on your requirements
    # 
    cropped_pil_image = pil_image.crop((100, 100, 400, 400))

    # Step 5: Display the cropped image
    plt.imshow(cropped_pil_image)
    plt.title("Cropped Image (Pillow)")
    plt.axis('off')
    plt.show()
cropped_image_path = './cropped_image.jpg'
cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))  # Save in BGR format
print(f"Cropped image saved as {cropped_image_path}")
