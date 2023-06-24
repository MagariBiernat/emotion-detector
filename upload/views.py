from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import cv2, os, urllib.request
import sys

face_detection_videocam = cv2.CascadeClassifier(
    os.path.join(
        settings.BASE_DIR, "opencv_haarcascade_data/haarcascade_frontalface_default.xml"
    )
)
# Create your views here.

media_root = settings.MEDIA_ROOT
from .opencv_face import opencv_face


def index(request, *args, **kwargs):
    if request.method == "POST" and request.FILES["myfile"]:
        if "image" in request.FILES["myfile"].content_type:
            myfile = request.FILES["myfile"]
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)

            opencv_face(media_root + "\\" + filename)

            return render(
                request, "upload/home.html", {"uploaded_file_url": uploaded_file_url}
            )
        else:
            return render(
                request,
                "upload/home.html",
                {"uploaded_file_url": "It has to be an image"},
            )

    return render(request, "upload/home.html")
