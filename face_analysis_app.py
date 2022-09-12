import streamlit as st
from PIL import Image, ImageDraw
import boto3
import base64
from io import BytesIO

st.set_page_config(page_title='Face😃 Analysis App')


def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href


def rekognition_detect_faces(photo):
    # API Call
    client = boto3.client('rekognition')
    photo_data = open(photo, 'rb')
    response = client.detect_faces(Image={'Bytes': photo_data.read()}, Attributes=['ALL'])
    # with open('tmp_face_analysis.json') as f:
    #     response = json.load(f)
    # with open('tmp_face_analysis.json', 'w') as f:
    #     json.dump(response, f)
    # print(response)
    return response


def process_image(bounding_boxes, img):
    image = Image.open(img)
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)
    for i in range(len(bounding_boxes)):
        width = img_width * bounding_boxes[i]['Width']
        height = img_height * bounding_boxes[i]['Height']
        left = img_width * bounding_boxes[i]['Left']
        top = img_height * bounding_boxes[i]['Top']
        points = ((left, top), (left + width, top), (left + width,
                                                     top + height), (left, top + height), (left, top))
        draw.line(points, fill='#00d400', width=4)
    return image


def run():
    st.title("Face😃 Analysis using AWS Rekognition")
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])
    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        st.image(save_image_path)
        if st.button("Analysis"):
            api_response = rekognition_detect_faces(save_image_path)
            bounding_box = []
            face_counter = 0
            if len(api_response['FaceDetails']) != 0:
                st.header(f"**Facial Analysis for: {img_file.name}**")
                st.success(f"Total {len(api_response['FaceDetails'])} Faces Detected")
                for face_detail in api_response['FaceDetails']:
                    bounding_box.append(face_detail['BoundingBox'])
                    face_counter += 1
                    st.markdown(
                        '''<h4 style='text-align: left; font-weight: bold; color: #1ed760;'>''' + f"Face - {face_counter} Analysis",
                        unsafe_allow_html=True)

                    # Age
                    st.subheader("Age Range")
                    st.info(f"Age Range: {face_detail['AgeRange']['Low']}-{face_detail['AgeRange']['High']} years old")

                    # Gender
                    st.subheader("Gender👦")
                    st.write(f"Gender: {face_detail['Gender']['Value']}")

                    # Emotion info
                    st.subheader("Emotion😊 Analysis")
                    st.write(f"Emotions: {face_detail['Emotions'][0]['Type']}")
                    st.write(f"Confidence: {int(face_detail['Emotions'][0]['Confidence'])}%")
                    st.progress(int(face_detail['Emotions'][0]['Confidence']))

                    # Smile Info
                    st.subheader("Smile😄 Detection")
                    st.write(f"Smile: {face_detail['Smile']['Value']} ")
                    st.write(f"Confidence: {int(face_detail['Smile']['Confidence'])}%")
                    st.progress(int(face_detail['Smile']['Confidence']))

                    # Eyes Open
                    st.subheader("Eyes👀 ")
                    st.write(f"Eyes Open: {face_detail['EyesOpen']['Value']} ")
                    st.write(f"Confidence: {int(face_detail['EyesOpen']['Confidence'])}%")
                    st.progress(int(face_detail['Smile']['Confidence']))

                    # Eyes Glasses
                    st.subheader("Eyes Glass 👓 ")
                    st.write(f"Eye Glasses: {face_detail['Eyeglasses']['Value']} ")
                    st.write(f"Confidence: {int(face_detail['Eyeglasses']['Confidence'])}%")
                    st.progress(int(face_detail['Smile']['Confidence']))

                pr_img = process_image(bounding_box, save_image_path)

                # Result Image
                st.subheader("Download Result")
                st.image(pr_img)

                # Download result
                st.markdown(get_image_download_link(pr_img, img_file.name, 'Download ' + img_file.name),
                            unsafe_allow_html=True)
            else:
                st.error("No Faces found!!")


run()
