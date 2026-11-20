from sklearn.datasets import fetch_lfw_people
import cv2

lfw = fetch_lfw_people(
    min_faces_per_person=10,
    color=True,
    resize=1.0
)

image = lfw.images[0]

print("Data type:", image.dtype)
print("Minimum pixel:", image.min())
print("Maximum pixel:", image.max())

img = cv2.imread("lfw_experiment/gallery/Abdullah Gul/registration.jpg")

print("\nSaved image:")
print("Minimum pixel:", img.min())
print("Maximum pixel:", img.max())