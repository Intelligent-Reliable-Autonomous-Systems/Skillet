from PIL import Image, ImageDraw

# Create a simple tabletop scene with objects to segment
img = Image.new("RGB", (640, 480), color=(200, 180, 150))  # wooden table background
draw = ImageDraw.Draw(img)

# Draw a tray
draw.rectangle([150, 280, 490, 400], fill=(180, 180, 180), outline=(100, 100, 100), width=4)

# Draw an apple
draw.ellipse([260, 200, 380, 310], fill=(200, 30, 30), outline=(150, 20, 20), width=3)
# Apple stem
draw.line([320, 200, 325, 175], fill=(80, 50, 20), width=4)

# Draw a cup
draw.rectangle([400, 230, 460, 340], fill=(70, 130, 180), outline=(50, 100, 150), width=3)

img.save("test_robot.jpg")
print("Saved test_robot.jpg")
