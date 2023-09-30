import streamlit as st
import tempfile
import base64
from pdf2image import convert_from_path
import pytesseract
import numpy as np
import pandas as pd
import pickle
import cv2
import spacy
import re
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)

nlp = spacy.load("en_core_web_sm")


def showpdf(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def convert_to_image(file_path):
    images = convert_from_path(file_path)
    return images


def get_image_txt(image):
    return pytesseract.image_to_string(image)


def resume_segmentation(images):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = np.array(images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    section_headings = []
    section_contents = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        text = get_image_txt(image[y : y + h, x : x + w])
        text = cleaned_text(text)

        matched_heading = next(
            (
                heading
                for heading in [
                    "EXPERIENCE",
                    "PROJECTS",
                    "EDUCATION",
                    "SKILLS",
                    "CERTIFICATIONS",
                ]
                if heading in text
            ),
            None,
        )
        if matched_heading:
            section_headings.append(matched_heading)
            section_contents.append(text.strip())
        elif section_headings:
            section_headings.append("Personal Details")
            section_contents.append(text.strip())
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    # cv2.imshow("image", resize_image(image))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    section_headings.reverse()
    section_contents.reverse()
    sections = dict(zip(section_headings, section_contents))
    return {"image": image, "sections": sections}


def resize_image(image):
    height, width = image.shape[:2]
    max_height = 800
    max_width = 800
    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def cleaned_text(text):
    text = re.sub(r"^[^\w\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r" +", " ", text, flags=re.MULTILINE)
    return text


def get_text_sections(image):
    return resume_segmentation(image)["sections"]


def get_experience(text):
    try:
        experience = ""
        for item in text.split("\n"):
            if (
                item.strip()
                and item.strip() != "EXPERIENCE"
                and item.strip() != "EXPERIENCE:"
            ):
                experience = experience + " " + item.strip()
        return experience
    except:
        pass


def get_educational_details(text):
    try:
        edu_details = ""
        for item in text.split("\n"):
            if (
                item.strip()
                and item.strip() != "EDUCATION"
                and item.strip() != "EDUCATION:"
            ):
                edu_details = edu_details + " " + item.strip()
        return edu_details
    except:
        pass


def get_project_details(text):
    try:
        doc = nlp(text)
        text_without_urls = " ".join(token.text for token in doc if not token.like_url)
        project_details = ""
        for item in text_without_urls.split("\n"):
            if item.strip() and (
                item.strip() != "PROJECTS" and item.strip() != "PROJECTS:"
            ):
                project_details = project_details + " " + item.strip()
        return project_details
    except:
        pass


def get_skills(text):
    try:
        doc = nlp(text)
        skills_list = [
            item.strip()
            for item in text.split("\n")
            if item.strip() and (item.strip() != "SKILLS" and item.strip() != "SKILLS:")
        ]
        return skills_list
    except:
        pass


def get_email(resume_text):
    try:
        EMAIL_REG = re.compile(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+")
        return re.findall(EMAIL_REG, resume_text)[0]
    except:
        pass


def get_name(txt):
    try:
        doc = nlp(txt)
        person_names = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = " ".join(ent.text.split())
                person_names.append(name)
        if not person_names:
            person_names.append(txt.split(" ")[0])

        return person_names[0]
    except:
        pass


def get_phone_number(resume_text):
    try:
        PHONE_REG = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
        phone = re.findall(PHONE_REG, resume_text)

        if phone:
            number = "".join(phone[0])

            if resume_text.find(number) >= 0 and len(number) < 13:
                return number
        return None
    except:
        pass


def get_resume_data(resume_sections):
    resume_data = {}
    personal_details = (
        resume_sections["Personal Details"]
        if "Personal Details" in resume_sections
        else None
    )
    if personal_details:
        resume_data["name"] = get_name(personal_details)
        resume_data["email"] = get_email(personal_details)
        resume_data["ph_num"] = get_phone_number(personal_details)

    experience_details = (
        get_experience(resume_sections["EXPERIENCE"])
        if "EXPERIENCE" in resume_sections
        else None
    )
    resume_data["experience"] = experience_details

    project_details = (
        get_project_details(resume_sections["PROJECTS"])
        if "PROJECTS" in resume_sections
        else None
    )
    educational_details = (
        get_educational_details(resume_sections["EDUCATION"])
        if "EDUCATION" in resume_sections
        else None
    )

    skills_details = (
        get_skills(resume_sections["SKILLS"]) if "SKILLS" in resume_sections else None
    )
    resume_data["skills"] = skills_details

    resume_data["projects"] = project_details
    resume_data["education"] = educational_details
    return resume_data


def summarize_resume(text):
    text = str(text)
    summarizer = pipeline("summarization", model="Samir001/ResumeSummary-t5-Wang-Arora")
    summarized_text = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summarized_text[0]["summary_text"]


def displayInfo(resume_data):
    if "name" in resume_data:
        st.title(f"Hi {resume_data['name']}")
        st.subheader("Your Basic Info", divider="rainbow")
        st.write(f"Name: {resume_data['name']}")
    if "email" in resume_data:
        st.write(f"Email: {resume_data['email']}")
    if "ph_num" in resume_data:
        st.write(f"Phone Number: {resume_data['ph_num']}")
    if "skills" in resume_data and resume_data["skills"]:
        st.subheader("Skills that you have", divider="rainbow")
        for i in resume_data["skills"]:
            st.markdown("- " + i)

    raw_text = get_raw_text(resume_data)
    processed_text = preprocess(raw_text)
    # st.write(raw_text)
    # print(resume_data.get("experience", ""))
    role = resume_role_classifier(processed_text)
    st.subheader(
        f"Based on your skills and experience, a career in {role} seems like a good fit for your resume"
    )


def preprocess(text):
    doc = nlp(text)
    no_stop_words = [
        token.text for token in doc if not token.is_stop and not token.is_punct
    ]
    return " ".join(no_stop_words)


def get_raw_text(resume_data):
    raw_text = "EXPERIENCE " + resume_data.get("experience", "")

    if resume_data.get("projects"):
        raw_text += " PROJECTS " + resume_data["projects"]

    if resume_data.get("education"):
        raw_text += " EDUCATION " + resume_data["education"]

    if resume_data.get("skills"):
        raw_text += " SKILLS " + " ".join(resume_data["skills"])

    return raw_text


def resume_role_classifier(text):
    label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
    role_classifier = pickle.load(open("model/resume_role_classifier_model.pkl", "rb"))
    input = pd.Series(text)
    pred = role_classifier.predict(input)
    decoded_prediction = label_encoder.inverse_transform(pred)
    return decoded_prediction[0]


def main():
    st.header("Resume Analyzer")
    st.title("Upload your Resume")
    file = st.file_uploader("upload your resume here", type=["pdf"])
    if file is not None:
        analyze = st.button("Analyze")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        showpdf(temp_file_path)  # uncomment later

        if analyze:
            image = convert_to_image(temp_file_path)[0]
            raw_text = get_image_txt(image)
            # st.write(raw_text)
            resume_sections = get_text_sections(image)
            # st.write(resume_sections)
            resume_data = get_resume_data(resume_sections)
            # st.write(resume_data)
            displayInfo(resume_data)
            summarized_text = summarize_resume(resume_data)
            st.subheader("Resume Summary", divider="rainbow")
            st.write(summarized_text)


if __name__ == "__main__":
    main()
