import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from main_file import getUniqueFramesList
import os
from flask import redirect, url_for, session, request
import os
import signal
import subprocess
import time
from flask import redirect, url_for, render_template_string
import subprocess



def delete_files_in_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    pass
                os.remove(file_path)
        except Exception as e:
            app.logger.error(f"Error deleting file: {e}")

def delete_jpg_files(folder_path):
    try:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a .JPG file
            if filename.lower().endswith(".jpg"):
                file_path = os.path.join(folder_path, filename)
                # Remove the file
                os.remove(file_path)
    except Exception as e:
        app.logger.error(f"Error deleting file: {e}")

@app.route('/')
def upload_form():
    delete_files_in_upload_folder()
    folder_path = app.config['IMAGES_FOLDER']
    delete_jpg_files(folder_path)
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    delete_files_in_upload_folder()
    folder_path = app.config['IMAGES_FOLDER']
    delete_jpg_files(folder_path)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        language = ""
        if 'language' in request.form:
            language = request.form['language']
        if language:
            app.logger.info(f"File Name ::: {filename}")
            final_list = getUniqueFramesList(language, filename)
            #final_list = [["aa", "bb", "cc"], ["1000", "20000", "30000"]]
            if len(final_list) == 2:
                if len(final_list[0]) >= 1 and len(final_list[1]) >= 1:
                    flash('Video successfully uploaded and displayed below')
                    contents = final_list[0]
                    timestamps = final_list[1]
                    return render_template('video_menu.html', filename=filename, contents=contents, timestamps=timestamps)
                else:
                    flash("There are no any Underlined Texts or Highlighted Texts or Extra Explanations in this video")
            else:
                flash("There are no any Underlined Texts or Highlighted Texts or Extra Explanations in this video")
    # If the code reaches here, it means the upload failed or there was no language specified
    flash("Upload failed or no language specified")
    return redirect(url_for('upload_form'))

@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
