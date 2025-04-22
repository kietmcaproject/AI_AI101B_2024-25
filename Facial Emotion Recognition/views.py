# from django.shortcuts import render
# from deepface import DeepFace

# def index(request):
#     emotion = None
#     if request.method == 'POST' and request.FILES.get('image'):
#         img = request.FILES['image']
#         path = f'media/{img.name}'
#         with open(path, 'wb+') as destination:
#             for chunk in img.chunks():
#                 destination.write(chunk)

#         try:
#             analysis = DeepFace.analyze(img_path=path, actions=['emotion'])
#             emotion = analysis[0]['dominant_emotion']
#         except Exception as e:
#             emotion = "Detection Error"

#     return render(request, 'index.html', {'emotion': emotion})

# from django.shortcuts import render
# from deepface import DeepFace


# def index(request):
#     emotion = None

#     if request.method == "POST" and request.FILES.get("image"):
#         image_file = request.FILES["image"]

#         # Save uploaded or webcam-captured image to a temp file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#             for chunk in image_file.chunks():
#                 tmp.write(chunk)
#             tmp_path = tmp.name

#         try:
#             analysis = DeepFace.analyze(img_path=tmp_path, actions=['emotion'], enforce_detection=False)
#             emotion = analysis[0]['dominant_emotion']
#         except Exception as e:
#             emotion = f"Error: {str(e)}"

#     return render(request, "index.html", {"emotion": emotion})

# from django.http import JsonResponse

# def index(request):
#     if request.method == "POST" and request.FILES.get("image"):
#         image_file = request.FILES["image"]

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#             for chunk in image_file.chunks():
#                 tmp.write(chunk)
#             tmp_path = tmp.name

#         try:
#             analysis = DeepFace.analyze(img_path=tmp_path, actions=['emotion'], enforce_detection=False)
#             emotion = analysis[0]['dominant_emotion']
#         except Exception as e:
#             emotion = f"Error: {str(e)}"

#         return JsonResponse({"emotion": emotion})

#     return render(request, "index.html")

from django.shortcuts import render
from django.http import JsonResponse
from deepface import DeepFace
from PIL import Image
import numpy as np
# import tempfile

def index(request):
    emotion = None

    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)

        try:
            result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        except Exception as e:
            emotion = "Error"

        # 👇 If it's an AJAX call (like from fetch in webcam mode), return JSON
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'emotion': emotion})
        else:
            # 👇 Otherwise render the page with result
            return render(request, 'index.html', {'emotion': emotion})

    return render(request, 'index.html')
